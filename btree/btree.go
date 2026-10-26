package btree

import (
	"bytes"
	"encoding/binary"
	"sort"
)

// PageSize is the size in bytes of every page read or allocated by PageStore.
const PageSize = 4096

const (
	maxKeySize     = 1000
	maxValueSize   = 3000
	nodeHeaderSize = 4
)

// Both node types use this layout:
//
//	Type (2B) | NumKeys (2B) | Children (n*8B) | Offsets (n*2B) | KV records
//
// Each record contains KLen (2B), VLen (2B), key bytes, and value bytes.
// Offsets are cumulative record ends relative to the KV area; its initial
// offset is implicitly zero. Internal entries contain a lower-bound key and
// a child page number, with an empty value. Leaf child references are unused.
type bnode []byte
type nodeType uint16

const nodeInternal nodeType = 1
const nodeLeaf nodeType = 2

func init() {
	assert(nodeHeaderSize+8+2+4+maxKeySize+maxValueSize <= PageSize,
		"the largest entry must fit in one page")
}

func newNode(ntype nodeType) bnode {
	page := make(bnode, PageSize)
	page.setHeader(ntype, 0)
	return page
}

// appendKV appends an entry to a leaf without searching for its position.
// The caller supplies keys in order and must split an oversized result before
// inserting again. The original node is unchanged.
func (b bnode) appendKV(key, value []byte) bnode {
	assert(b.kind() == nodeLeaf, "append must be called only on leaf nodes")
	assert(b.size() <= PageSize, "split the oversized node before appending")
	dst := make(bnode, 2*PageSize)
	leafInsert(dst, b, b.numKeys(), key, value)
	if dst.size() <= PageSize {
		return dst[:PageSize]
	}
	return dst
}

func (b bnode) kind() nodeType {
	return nodeType(binary.LittleEndian.Uint16(b[:2]))
}

func (b bnode) numKeys() uint16 {
	return binary.LittleEndian.Uint16(b[2:nodeHeaderSize])
}

// setHeader fixes the entry count before constructing the variable-size arrays.
func (b bnode) setHeader(ntype nodeType, count uint16) {
	assert(ntype == nodeLeaf || ntype == nodeInternal, "invalid node type")
	assert(nodeHeaderSize+10*int(count) <= len(b), "node metadata exceeds buffer")
	binary.LittleEndian.PutUint16(b[:2], uint16(ntype))
	binary.LittleEndian.PutUint16(b[2:nodeHeaderSize], count)
}

func (b bnode) firstOffset() uint16 {
	return nodeHeaderSize + 8*b.numKeys()
}

// getOffset accepts count as well as entry indices: getOffset(count) is the
// length of the kv area, while getOffset(0) is its implicit starting boundary.
func (b bnode) getOffset(pos uint16) uint16 {
	assert(pos <= b.numKeys(), "offset index out of range")
	if pos == 0 {
		return 0
	}
	idx := b.firstOffset() + 2*(pos-1)
	return binary.LittleEndian.Uint16(b[idx : idx+2])
}

func (b bnode) setOffset(pos, offset uint16) {
	assert(pos > 0 && pos <= b.numKeys(), "stored offset index out of range")
	idx := b.firstOffset() + 2*(pos-1)
	binary.LittleEndian.PutUint16(b[idx:idx+2], offset)
}

func (b bnode) kvPos(pos uint16) uint16 {
	return b.firstOffset() + 2*b.numKeys() + b.getOffset(pos)
}

func (b bnode) key(pos uint16) []byte {
	assert(pos < b.numKeys(), "key index out of range")
	start := b.kvPos(pos)
	length := binary.LittleEndian.Uint16(b[start : start+2])
	return b[start+4 : start+4+length]
}

func (b bnode) value(pos uint16) []byte {
	_, value := b.kv(pos)
	return value
}

func (b bnode) kv(pos uint16) ([]byte, []byte) {
	assert(pos < b.numKeys(), "entry index out of range")
	start := b.kvPos(pos)
	keyLen := binary.LittleEndian.Uint16(b[start : start+2])
	valLen := binary.LittleEndian.Uint16(b[start+2 : start+4])
	keyStart := start + 4
	valStart := keyStart + keyLen
	return b[keyStart:valStart], b[valStart : valStart+valLen]
}

// child takes an entry index and returns a page number, not a byte address.
func (b bnode) child(pos int64) uint64 {
	assert(pos >= 0 && pos < int64(b.numKeys()), "child index out of range")
	idx := nodeHeaderSize + 8*pos
	return binary.LittleEndian.Uint64(b[idx : idx+8])
}

func (b bnode) setPtr(pos uint16, page uint64) {
	assert(pos < b.numKeys(), "child index out of range")
	idx := nodeHeaderSize + 8*int(pos)
	binary.LittleEndian.PutUint64(b[idx:idx+8], page)
}

func (b bnode) size() uint16 {
	return b.kvPos(b.numKeys())
}

// nodeAppendKV writes the next entry in a node being constructed. The final
// header must already be set, and entries must be written in increasing order.
func nodeAppendKV(dst bnode, pos uint16, child uint64, key, value []byte) {
	assert(pos < dst.numKeys(), "destination entry index out of range")
	assert(len(key) <= maxKeySize, "key exceeds size limit")
	assert(len(value) <= maxValueSize, "value exceeds size limit")
	assert(dst.kind() != nodeInternal || len(value) == 0,
		"internal entries cannot store values")
	start := int(dst.kvPos(pos))
	keyStart := start + 4
	valStart := keyStart + len(key)
	end := valStart + len(value)
	assert(end <= len(dst), "entry exceeds node buffer")
	dst.setPtr(pos, child)
	binary.LittleEndian.PutUint16(dst[start:start+2], uint16(len(key)))
	binary.LittleEndian.PutUint16(dst[start+2:start+4], uint16(len(value)))
	copy(dst[keyStart:valStart], key)
	copy(dst[valStart:end], value)
	dst.setOffset(pos+1, dst.getOffset(pos)+uint16(4+len(key)+len(value)))
}

func nodeAppendRange(dst, src bnode, dstPos, srcPos, count uint16) {
	assert(dst.kind() == src.kind(), "cannot copy entries between different node types")
	assert(int(dstPos)+int(count) <= int(dst.numKeys()), "destination range out of bounds")
	assert(int(srcPos)+int(count) <= int(src.numKeys()), "source range out of bounds")
	for i := uint16(0); i < count; i++ {
		key, value := src.kv(srcPos + i)
		nodeAppendKV(dst, dstPos+i, src.child(int64(srcPos+i)), key, value)
	}
}

// nodeLookupLE finds the greatest key <= key. The node must be nonempty and
// cover the requested key; its first lower bound must not exceed key.
func nodeLookupLE(node bnode, key []byte) uint16 {
	end := sort.Search(int(node.numKeys()), func(i int) bool {
		return bytes.Compare(node.key(uint16(i)), key) > 0
	})
	assert(end > 0, "key is outside the node's range")
	return uint16(end - 1)
}

func leafInsert(dst, src bnode, pos uint16, key, value []byte) {
	assert(src.kind() == nodeLeaf, "insertion requires a leaf")
	assert(pos <= src.numKeys(), "insertion index out of range")
	dst.setHeader(nodeLeaf, src.numKeys()+1)
	nodeAppendRange(dst, src, 0, 0, pos)
	nodeAppendKV(dst, pos, 0, key, value)
	nodeAppendRange(dst, src, pos+1, pos, src.numKeys()-pos)
}

func leafUpdate(dst, src bnode, pos uint16, key, value []byte) {
	assert(src.kind() == nodeLeaf, "update requires a leaf")
	assert(pos < src.numKeys(), "update index out of range")
	assert(bytes.Equal(src.key(pos), key), "update must preserve the existing key")
	dst.setHeader(nodeLeaf, src.numKeys())
	nodeAppendRange(dst, src, 0, 0, pos)
	nodeAppendKV(dst, pos, 0, key, value)
	nodeAppendRange(dst, src, pos+1, pos+1, src.numKeys()-pos-1)
}

// rangeSize is the encoded size of a standalone node containing [start, end).
func (b bnode) rangeSize(start, end uint16) int {
	return nodeHeaderSize + 10*int(end-start) + int(b.getOffset(end)-b.getOffset(start))
}

// nodeSplit2 produces two nonempty nodes. The right node fits in one page;
// uneven record sizes can leave the left node in need of another split.
func nodeSplit2(left, right, src bnode) {
	count := src.numKeys()
	assert(count >= 2, "splitting requires at least two entries")
	cut := count / 2
	for src.rangeSize(0, cut) > PageSize {
		cut--
	}
	for src.rangeSize(cut, count) > PageSize {
		cut++
	}
	assert(cut > 0 && cut < count, "split must leave both nodes nonempty")
	left.setHeader(src.kind(), cut)
	right.setHeader(src.kind(), count-cut)
	nodeAppendRange(left, src, 0, 0, cut)
	nodeAppendRange(right, src, 0, cut, count-cut)
	assert(right.size() <= PageSize, "right split exceeds page size")
}

// nodeSplit3 returns one, two, or three page-sized nodes in key order.
// An overflowing input is a temporary node of at most two pages.
func nodeSplit3(src bnode) (uint16, [3]bnode) {
	assert(src.size() <= 2*PageSize, "temporary node exceeds two pages")
	if src.size() <= PageSize {
		return 1, [3]bnode{src[:PageSize]}
	}
	left := make(bnode, 2*PageSize)
	right := newNode(src.kind())
	nodeSplit2(left, right, src)
	if left.size() <= PageSize {
		return 2, [3]bnode{left[:PageSize], right}
	}
	first, middle := newNode(src.kind()), newNode(src.kind())
	nodeSplit2(first, middle, left)
	assert(first.size() <= PageSize, "left split exceeds page size")
	return 3, [3]bnode{first, middle, right}
}

func assert(condition bool, message string) {
	if !condition {
		panic(message)
	}
}
