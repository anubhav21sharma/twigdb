package btree

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

type testPages struct {
	t     *testing.T
	pages map[uint64]bnode
	next  uint64
	free  []uint64
}

func testTree(t *testing.T) (*Tree, *testPages) {
	t.Helper()
	store := &testPages{t: t, pages: make(map[uint64]bnode), next: 1}
	return New(0, store), store
}

func (s *testPages) ReadPage(page uint64) []byte {
	s.t.Helper()
	node, ok := s.pages[page]
	if !ok {
		s.t.Fatalf("fetch of missing page %d", page)
	}
	return node
}

func (s *testPages) AllocatePage(data []byte) uint64 {
	s.t.Helper()
	if len(data) != PageSize || bnode(data).size() > PageSize {
		s.t.Fatal("attempted to persist an invalid page")
	}
	var page uint64
	if len(s.free) > 0 {
		page = s.free[len(s.free)-1]
		s.free = s.free[:len(s.free)-1]
	} else {
		page = s.next
		s.next++
	}
	s.pages[page] = bnode(bytes.Clone(data))
	return page
}

func (s *testPages) ReleasePage(page uint64) {
	s.t.Helper()
	if _, ok := s.pages[page]; !ok {
		s.t.Fatalf("release of missing page %d", page)
	}
	delete(s.pages, page)
	s.free = append(s.free, page)
}

// Compare persisted structure and public lookups with an independent map.
func assertTree(t *testing.T, tree *Tree, store *testPages, want map[string]string) int {
	t.Helper()
	if tree.Root() == 0 {
		if len(want) != 0 || len(store.pages) != 0 {
			t.Fatal("rootless tree contains data or unreachable pages")
		}
		return 0
	}
	seen := make(map[uint64]bool)
	values := make(map[string]string)
	var walk func(uint64, []byte, []byte) int
	walk = func(page uint64, lower, upper []byte) int {
		if seen[page] {
			t.Fatalf("page %d is referenced more than once", page)
		}
		seen[page] = true
		node := bnode(store.ReadPage(page))
		if node.numKeys() == 0 || node.size() > PageSize || !bytes.Equal(node.key(0), lower) {
			t.Fatalf("invalid page or lower bound at page %d", page)
		}
		height := 0
		for i := uint16(0); i < node.numKeys(); i++ {
			key, value := node.kv(i)
			if (i > 0 && bytes.Compare(node.key(i-1), key) >= 0) ||
				bytes.Compare(key, lower) < 0 || (upper != nil && bytes.Compare(key, upper) >= 0) {
				t.Fatalf("unsorted or out-of-range key on page %d", page)
			}
			switch node.kind() {
			case nodeLeaf:
				if len(key) > 0 {
					values[string(key)] = string(value)
				} else if len(value) != 0 {
					t.Fatal("sentinel has a value")
				}
			case nodeInternal:
				if len(value) != 0 {
					t.Fatal("internal entry has a value")
				}
				next := upper
				if i+1 < node.numKeys() {
					next = node.key(i + 1)
				}
				childHeight := walk(node.child(int64(i)), key, next)
				if i > 0 && height != childHeight {
					t.Fatal("leaves have different depths")
				}
				height = childHeight
			default:
				t.Fatal("invalid node type")
			}
		}
		return height + 1
	}
	height := walk(tree.Root(), nil, nil)
	if len(seen) != len(store.pages) || len(values) != len(want) {
		t.Fatalf("page leak or wrong key count: pages=%d/%d, keys=%d/%d",
			len(seen), len(store.pages), len(values), len(want))
	}
	for key, expected := range want {
		if value, ok := values[key]; !ok || value != expected {
			t.Fatalf("stored value for %q = %q (present=%v), want %q", key, value, ok, expected)
		}
		if value, ok := tree.Get([]byte(key)); !ok || string(value) != expected {
			t.Fatalf("Get(%q) = %q, %v; want %q", key, value, ok, expected)
		}
	}
	return height
}

func TestTreeAPI(t *testing.T) {
	tree, store := testTree(t)
	if value, ok := tree.Get([]byte("missing")); ok || value != nil {
		t.Fatal("empty tree returned a value")
	}
	if deleted, err := tree.Delete([]byte("missing")); deleted || err != nil {
		t.Fatalf("empty delete = %v, %v", deleted, err)
	}
	for _, entry := range []struct{ key, value string }{{"key", "old"}, {"key", "updated"}, {"empty", ""}} {
		if err := tree.Insert([]byte(entry.key), []byte(entry.value)); err != nil {
			t.Fatal(err)
		}
	}
	tree = New(tree.Root(), store)
	value, ok := tree.Get([]byte("key"))
	if !ok || string(value) != "updated" {
		t.Fatalf("reopened lookup = %q, %v", value, ok)
	}
	value[0] = 'X'
	want := map[string]string{"key": "updated", "empty": ""}
	assertTree(t, tree, store, want)
	root, count := tree.Root(), len(store.pages)
	for _, invalid := range []struct{ key, value []byte }{
		{nil, nil},
		{make([]byte, maxKeySize+1), nil},
		{[]byte("key"), make([]byte, maxValueSize+1)},
	} {
		if err := tree.Insert(invalid.key, invalid.value); err == nil {
			t.Fatal("invalid insert was accepted")
		}
		if tree.Root() != root || len(store.pages) != count {
			t.Fatal("invalid insert changed storage")
		}
	}
	for _, key := range [][]byte{nil, make([]byte, maxKeySize+1)} {
		if deleted, err := tree.Delete(key); deleted || err == nil {
			t.Fatal("invalid delete was accepted")
		}
		if _, ok := tree.Get(key); ok {
			t.Fatal("invalid lookup exposed an entry")
		}
	}
	if deleted, err := tree.Delete([]byte("missing")); deleted || err != nil {
		t.Fatalf("missing delete = %v, %v", deleted, err)
	}
	if tree.Root() != root || len(store.pages) != count {
		t.Fatal("invalid or missing delete changed storage")
	}
	assertTree(t, tree, store, want)
}

func TestTreeLifecycle(t *testing.T) {
	for _, order := range []string{"ascending", "descending", "random"} {
		t.Run(order, func(t *testing.T) {
			tree, store := testTree(t)
			keys := make([][]byte, 60)
			want := make(map[string]string)
			for i := range keys {
				keys[i] = append([]byte(fmt.Sprintf("%04d", i)), bytes.Repeat([]byte{'k'}, maxKeySize-4)...)
			}
			for _, i := range rand.New(rand.NewSource(9)).Perm(len(keys)) {
				value := bytes.Repeat([]byte{byte(i)}, maxValueSize)
				if err := tree.Insert(keys[i], value); err != nil {
					t.Fatal(err)
				}
				want[string(keys[i])] = string(value)
			}
			if assertTree(t, tree, store, want) < 3 {
				t.Fatal("fixture did not exercise internal splits")
			}
			oldRoot := store.pages[tree.Root()]
			before := bytes.Clone(oldRoot)
			if err := tree.Insert(keys[0], []byte("updated")); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(oldRoot, before) {
				t.Fatal("update mutated the original root")
			}
			want[string(keys[0])] = "updated"
			indices := rand.New(rand.NewSource(17)).Perm(len(keys))
			for i := range indices {
				if order == "ascending" {
					indices[i] = i
				} else if order == "descending" {
					indices[i] = len(keys) - i - 1
				}
			}
			for _, i := range indices {
				if deleted, err := tree.Delete(keys[i]); err != nil || !deleted {
					t.Fatalf("delete = %v, %v", deleted, err)
				}
				delete(want, string(keys[i]))
				assertTree(t, tree, store, want)
				if _, ok := tree.Get(keys[i]); ok {
					t.Fatal("deleted key is still present")
				}
			}
			if len(store.pages) != 1 || bnode(tree.pages.ReadPage(tree.Root())).kind() != nodeLeaf {
				t.Fatal("drained tree should retain only its sentinel leaf")
			}
			if err := tree.Insert([]byte{0}, []byte("reinserted")); err != nil {
				t.Fatal(err)
			}
			assertTree(t, tree, store, map[string]string{string([]byte{0}): "reinserted"})
		})
	}
}

func TestTreeMixedOperations(t *testing.T) {
	for _, seed := range []int64{7, 42} {
		t.Run(fmt.Sprint(seed), func(t *testing.T) {
			tree, store := testTree(t)
			var keys [][]byte
			for i := 0; i < 80; i++ {
				key := []byte(fmt.Sprintf("%03d", i))
				if i%3 == 0 {
					key = append(key, bytes.Repeat([]byte{'k'}, maxKeySize-len(key))...)
				} else if i%3 == 1 {
					key = []byte{byte(i), 0, 255}
				}
				keys = append(keys, key)
			}
			want := make(map[string]string)
			rng := rand.New(rand.NewSource(seed))
			for step := 0; step < 800; step++ {
				key := keys[rng.Intn(len(keys))]
				if rng.Intn(5) < 3 {
					value := bytes.Repeat([]byte{byte(rng.Intn(256))}, rng.Intn(maxValueSize+1))
					if err := tree.Insert(key, value); err != nil {
						t.Fatal(err)
					}
					want[string(key)] = string(value)
				} else {
					_, present := want[string(key)]
					root := tree.Root()
					if deleted, err := tree.Delete(key); err != nil || deleted != present {
						t.Fatalf("step %d: delete = %v, %v; present=%v", step, deleted, err, present)
					}
					if !present && tree.Root() != root {
						t.Fatal("missing delete replaced the root")
					}
					delete(want, string(key))
				}
				assertTree(t, tree, store, want)
				expected, present := want[string(key)]
				if value, ok := tree.Get(key); ok != present || (present && string(value) != expected) {
					t.Fatalf("lookup after step %d = %q, %v", step, value, ok)
				}
			}
		})
	}
}

func TestDeleteLongerRoutingKey(t *testing.T) {
	for _, wrap := range []bool{false, true} {
		t.Run(fmt.Sprint(wrap), func(t *testing.T) {
			tree, store := testTree(t)
			want := map[string]string{"a": "old"}
			longKey := bytes.Repeat([]byte{'a'}, maxKeySize)
			largeValue := bytes.Repeat([]byte{'v'}, 2000)
			left := store.AllocatePage(testNode(nodeLeaf, testEntry{}))
			target := store.AllocatePage(testNode(nodeLeaf,
				testEntry{key: []byte("a"), value: []byte("old")},
				testEntry{key: longKey, value: largeValue},
			))
			want[string(longKey)] = string(largeValue)
			entries := []testEntry{{child: left}, {key: []byte("a"), child: target}}
			for _, first := range []byte{'b', 'c', 'd', 'e'} {
				key := bytes.Repeat([]byte{first}, maxKeySize)
				child := store.AllocatePage(testNode(nodeLeaf, testEntry{key: key, value: largeValue}))
				entries = append(entries, testEntry{key: key, child: child})
				want[string(key)] = string(largeValue)
			}
			tree.root = store.AllocatePage(testNode(nodeInternal, entries...))
			if wrap {
				tree.root = store.AllocatePage(testNode(nodeInternal, testEntry{child: tree.root}))
			}
			assertTree(t, tree, store, want)
			deleted, err := tree.Delete([]byte("a"))
			if err != nil || !deleted {
				t.Fatalf("delete = %v, %v", deleted, err)
			}
			delete(want, "a")
			assertTree(t, tree, store, want)
			if value, ok := tree.Get(longKey); !ok || !bytes.Equal(value, largeValue) {
				t.Fatal("deletion lost the next, longer key")
			}
		})
	}
}
