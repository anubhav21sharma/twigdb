package btree

import (
	"bytes"
	"fmt"
)

// PageStore supplies the opaque pages used by a Tree. The KV layer owns the
// store's lifetime and saves the root page number returned by Root.
// Operations must complete successfully; the tree does not perform disk I/O.
type PageStore interface {
	// ReadPage returns PageSize read-only bytes, valid until the page is released.
	ReadPage(page uint64) []byte
	// AllocatePage stores PageSize bytes without modifying them and returns a
	// nonzero page number that is not already in use.
	AllocatePage(data []byte) uint64
	// ReleasePage releases a page number so the store may reuse it.
	ReleasePage(page uint64)
}

// Tree is a B+tree with lexicographically ordered byte keys. Its node format
// and balancing operations are private. Create a Tree with New.
type Tree struct {
	root  uint64
	pages PageStore
}

// New opens a tree on pages. Pass root=0 for a new tree, or a saved root page
// number to reopen one. The caller retains ownership of pages.
func New(root uint64, pages PageStore) *Tree {
	assert(pages != nil, "page storage is required")
	return &Tree{root: root, pages: pages}
}

// Root returns the current root page number. The storage owner is responsible
// for saving it after successful updates; zero means no root has been allocated.
func (tree *Tree) Root() uint64 {
	return tree.root
}

func checkLimit(key, value []byte) error {
	if len(key) == 0 {
		return fmt.Errorf("key must not be empty")
	}
	if len(key) > maxKeySize {
		return fmt.Errorf("key length %d exceeds limit %d", len(key), maxKeySize)
	}
	if len(value) > maxValueSize {
		return fmt.Errorf("value length %d exceeds limit %d", len(value), maxValueSize)
	}
	return nil
}

// Insert adds a key or replaces its value. Keys must contain 1 to 1000 bytes;
// values may contain 0 to 3000 bytes. Invalid lengths leave the tree unchanged.
func (tree *Tree) Insert(key, value []byte) error {
	if err := checkLimit(key, value); err != nil {
		return err
	}
	if tree.root == 0 {
		root := newNode(nodeLeaf)
		root.setHeader(nodeLeaf, 2)
		nodeAppendKV(root, 0, 0, nil, nil)
		nodeAppendKV(root, 1, 0, key, value)
		tree.root = tree.storeNode(root)
		return nil
	}
	updated := treeInsert(tree, bnode(tree.pages.ReadPage(tree.root)), key, value)
	tree.installRoot(updated)
	return nil
}

// Get returns an owned copy of the value and whether the key exists. Missing
// and invalid keys return (nil, false); an empty value can still be present.
func (tree *Tree) Get(key []byte) ([]byte, bool) {
	if tree.root == 0 || len(key) == 0 || len(key) > maxKeySize {
		return nil, false
	}
	page := tree.root
	for {
		node := bnode(tree.pages.ReadPage(page))
		pos := nodeLookupLE(node, key)
		switch node.kind() {
		case nodeLeaf:
			if !bytes.Equal(node.key(pos), key) {
				return nil, false
			}
			return bytes.Clone(node.value(pos)), true
		case nodeInternal:
			page = node.child(int64(pos))
		default:
			panic("invalid node type")
		}
	}
}

// Delete removes a key and reports whether it existed. A missing key leaves
// the root and stored pages unchanged. Invalid keys return an error.
func (tree *Tree) Delete(key []byte) (bool, error) {
	if err := checkLimit(key, nil); err != nil {
		return false, err
	}
	if tree.root == 0 {
		return false, nil
	}
	updated := treeDelete(tree, bnode(tree.pages.ReadPage(tree.root)), key)
	if len(updated) == 0 {
		return false, nil
	}
	tree.installRoot(updated)
	return true, nil
}

func (tree *Tree) storeNode(node bnode) uint64 {
	assert(node.size() <= PageSize, "cannot store an oversized node")
	page := tree.pages.AllocatePage(node[:PageSize])
	assert(page != 0, "page allocator must return a nonzero page number")
	return page
}

// installRoot handles growth as well as collapsing a root with one child.
// A deletion can grow routing keys, so it can also require a root split.
func (tree *Tree) installRoot(updated bnode) {
	previous := tree.root
	if updated.numKeys() == 0 {
		tree.root = 0
		tree.pages.ReleasePage(previous)
		return
	}
	count, nodes := nodeSplit3(updated)
	switch {
	case count > 1:
		root := newNode(nodeInternal)
		root.setHeader(nodeInternal, count)
		for i, child := range nodes[:count] {
			nodeAppendKV(root, uint16(i), tree.storeNode(child), child.key(0), nil)
		}
		tree.root = tree.storeNode(root)
	case nodes[0].kind() == nodeInternal && nodes[0].numKeys() == 1:
		page := nodes[0].child(0)
		for {
			node := bnode(tree.pages.ReadPage(page))
			if node.kind() != nodeInternal || node.numKeys() != 1 {
				break
			}
			next := node.child(0)
			tree.pages.ReleasePage(page)
			page = next
		}
		tree.root = page
	default:
		tree.root = tree.storeNode(nodes[0])
	}
	tree.pages.ReleasePage(previous)
}

// treeInsert returns a replacement for an existing subtree root. Its result
// can overflow a page; the caller handles splitting and replacing that root.
// Internal calls install split children and release their old page references.
func treeInsert(tree *Tree, node bnode, key, value []byte) bnode {
	assert(node.size() <= PageSize, "insert requires a page-sized source node")
	dst := make(bnode, 2*PageSize)
	if node.kind() == nodeLeaf && node.numKeys() == 0 {
		leafInsert(dst, node, 0, key, value)
		return dst
	}
	pos := nodeLookupLE(node, key)
	switch node.kind() {
	case nodeLeaf:
		if bytes.Equal(node.key(pos), key) {
			leafUpdate(dst, node, pos, key, value)
		} else {
			leafInsert(dst, node, pos+1, key, value)
		}
	case nodeInternal:
		page := node.child(int64(pos))
		child := treeInsert(tree, bnode(tree.pages.ReadPage(page)), key, value)
		count, children := nodeSplit3(child)
		tree.pages.ReleasePage(page)
		nodeReplaceKidN(tree, dst, node, pos, children[:count]...)
	default:
		panic("invalid node type")
	}
	return dst
}

// nodeReplaceKidN keeps unaffected child page numbers and replaces one child
// entry with the newly allocated children, using each child's first key.
func nodeReplaceKidN(tree *Tree, dst, src bnode, pos uint16, children ...bnode) {
	assert(src.kind() == nodeInternal, "child replacement requires an internal node")
	assert(pos < src.numKeys(), "child replacement index out of range")
	assert(len(children) > 0, "child replacement requires at least one child")
	count := uint16(len(children))
	dst.setHeader(nodeInternal, src.numKeys()-1+count)
	nodeAppendRange(dst, src, 0, 0, pos)
	for i, child := range children {
		assert(child.numKeys() > 0, "replacement child must be nonempty")
		page := tree.storeNode(child)
		nodeAppendKV(dst, pos+uint16(i), page, child.key(0), nil)
	}
	nodeAppendRange(dst, src, pos+count, pos+1, src.numKeys()-pos-1)
}

func leafDelete(dst, src bnode, pos uint16) {
	assert(src.kind() == nodeLeaf, "deletion requires a leaf")
	assert(pos < src.numKeys(), "deletion index out of range")
	dst.setHeader(nodeLeaf, src.numKeys()-1)
	nodeAppendRange(dst, src, 0, 0, pos)
	nodeAppendRange(dst, src, pos, pos+1, src.numKeys()-pos-1)
}

func nodeMerge(dst, left, right bnode) {
	assert(left.kind() == right.kind(), "cannot merge different node types")
	assert(int(left.size())+int(right.size())-nodeHeaderSize <= PageSize,
		"merged node exceeds one page")
	dst.setHeader(left.kind(), left.numKeys()+right.numKeys())
	nodeAppendRange(dst, left, 0, 0, left.numKeys())
	nodeAppendRange(dst, right, left.numKeys(), 0, right.numKeys())
}

func nodeReplace2Kid(dst, src bnode, pos uint16, page uint64, key []byte) {
	assert(src.kind() == nodeInternal, "child replacement requires an internal node")
	assert(int(pos)+1 < int(src.numKeys()), "two adjacent children are required")
	assert(page != 0, "child page number must be nonzero")
	dst.setHeader(nodeInternal, src.numKeys()-1)
	nodeAppendRange(dst, src, 0, 0, pos)
	nodeAppendKV(dst, pos, page, key, nil)
	nodeAppendRange(dst, src, pos+1, pos+2, src.numKeys()-pos-2)
}

// shouldMerge prefers the left sibling when a node is at most one-quarter
// full. Merging is optional unless an empty node has an available sibling.
func shouldMerge(tree *Tree, parent bnode, pos uint16, updated bnode) (int, bnode) {
	if updated.size() > PageSize/4 {
		return 0, nil
	}
	if pos > 0 {
		left := bnode(tree.pages.ReadPage(parent.child(int64(pos - 1))))
		if int(left.size())+int(updated.size())-nodeHeaderSize <= PageSize {
			return -1, left
		}
	}
	if pos+1 < parent.numKeys() {
		right := bnode(tree.pages.ReadPage(parent.child(int64(pos + 1))))
		if int(updated.size())+int(right.size())-nodeHeaderSize <= PageSize {
			return 1, right
		}
	}
	return 0, nil
}

// treeDelete returns nil when absent, or a replacement subtree root on success.
func treeDelete(tree *Tree, node bnode, key []byte) bnode {
	if node.numKeys() == 0 {
		return nil
	}
	pos := nodeLookupLE(node, key)
	switch node.kind() {
	case nodeLeaf:
		if !bytes.Equal(node.key(pos), key) {
			return nil
		}
		dst := newNode(nodeLeaf)
		leafDelete(dst, node, pos)
		return dst
	case nodeInternal:
		return nodeDelete(tree, node, pos, key)
	default:
		panic("invalid node type")
	}
}

func nodeDelete(tree *Tree, parent bnode, pos uint16, key []byte) bnode {
	page := parent.child(int64(pos))
	updated := treeDelete(tree, bnode(tree.pages.ReadPage(page)), key)
	if len(updated) == 0 {
		return nil
	}
	tree.pages.ReleasePage(page)
	// A new first key can be longer than the deleted key, growing the parent
	// even when its entry count stays the same. Allow that temporary overflow.
	dst := make(bnode, 2*PageSize)
	if updated.size() > PageSize {
		count, children := nodeSplit3(updated)
		nodeReplaceKidN(tree, dst, parent, pos, children[:count]...)
		return dst[:max(PageSize, int(dst.size()))]
	}
	direction, sibling := shouldMerge(tree, parent, pos, updated)
	switch {
	case direction < 0:
		merged := newNode(updated.kind())
		nodeMerge(merged, sibling, updated)
		tree.pages.ReleasePage(parent.child(int64(pos - 1)))
		nodeReplace2Kid(dst, parent, pos-1, tree.storeNode(merged), merged.key(0))
	case direction > 0:
		merged := newNode(updated.kind())
		nodeMerge(merged, updated, sibling)
		tree.pages.ReleasePage(parent.child(int64(pos + 1)))
		nodeReplace2Kid(dst, parent, pos, tree.storeNode(merged), merged.key(0))
	case updated.numKeys() == 0:
		assert(parent.numKeys() == 1 && pos == 0, "empty child must have no siblings")
		dst.setHeader(nodeInternal, 0)
	default:
		nodeReplaceKidN(tree, dst, parent, pos, updated)
	}
	return dst[:max(PageSize, int(dst.size()))]
}
