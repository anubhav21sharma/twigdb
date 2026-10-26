package btree

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

func TestAppendKV(t *testing.T) {
	entries := []struct {
		key string
		val string
	}{
		{"k1", "v1"},
		{"k2", "v2"},
	}

	bnode := newNode(nodeLeaf)
	if got := bnode.kind(); got != nodeLeaf {
		t.Fatalf("new node type = %v, want %v", got, nodeLeaf)
	}
	if got := bnode.numKeys(); got != 0 {
		t.Fatalf("new node keys = %d, want 0", got)
	}

	for i, e := range entries {
		bnode = bnode.appendKV([]byte(e.key), []byte(e.val))
		pos := uint16(i)

		if got := bnode.kind(); got != nodeLeaf {
			t.Fatalf("append %d: type = %v, want %v", i, got, nodeLeaf)
		}
		if got, want := bnode.numKeys(), uint16(i+1); got != want {
			t.Fatalf("append %d: keys = %d, want %d", i, got, want)
		}
		if got := string(bnode.key(pos)); got != e.key {
			t.Errorf("append %d: key(%d) = %q, want %q", i, pos, got, e.key)
		}
		if got := string(bnode.value(pos)); got != e.val {
			t.Errorf("append %d: value(%d) = %q, want %q", i, pos, got, e.val)
		}
		if k, v := bnode.kv(pos); string(k) != e.key || string(v) != e.val {
			t.Errorf("append %d: kv(%d) = %q,%q, want %q,%q", i, pos, k, v, e.key, e.val)
		}
	}
}

type testEntry struct {
	key, value []byte
	child      uint64
}

func testNode(ntype nodeType, entries ...testEntry) bnode {
	node := make(bnode, 2*PageSize)
	node.setHeader(ntype, uint16(len(entries)))
	for i, entry := range entries {
		nodeAppendKV(node, uint16(i), entry.child, entry.key, entry.value)
	}
	if node.size() <= PageSize {
		return node[:PageSize]
	}
	return node
}

func assertEntries(t *testing.T, node bnode, entries []testEntry) {
	t.Helper()
	if int(node.numKeys()) != len(entries) {
		t.Fatalf("entry count = %d, want %d", node.numKeys(), len(entries))
	}
	dataSize := 0
	for i, entry := range entries {
		key, value := node.kv(uint16(i))
		if !bytes.Equal(key, entry.key) || !bytes.Equal(value, entry.value) {
			t.Fatalf("entry %d = (%q, %q), want (%q, %q)", i, key, value, entry.key, entry.value)
		}
		if !bytes.Equal(node.key(uint16(i)), entry.key) || !bytes.Equal(node.value(uint16(i)), entry.value) {
			t.Fatalf("separate key/value getters disagree for entry %d", i)
		}
		if got := node.child(int64(i)); got != entry.child {
			t.Fatalf("child %d = %d, want %d", i, got, entry.child)
		}
		dataSize += 4 + len(entry.key) + len(entry.value)
		slot := 4 + 8*len(entries) + 2*i
		if got := int(binary.LittleEndian.Uint16(node[slot : slot+2])); got != dataSize {
			t.Fatalf("stored offset %d = %d, want %d", i, got, dataSize)
		}
	}
	if want := 4 + 10*len(entries) + dataSize; int(node.size()) != want {
		t.Fatalf("used size = %d, want %d", node.size(), want)
	}
}

func TestNodeEncoding(t *testing.T) {
	entries := []testEntry{
		{key: []byte("k1"), value: []byte("hi")},
		{key: []byte("k3"), value: []byte("hello")},
	}
	node := testNode(nodeLeaf, entries...)
	want := []byte{
		2, 0, 2, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		8, 0, 19, 0,
		2, 0, 2, 0, 'k', '1', 'h', 'i',
		2, 0, 5, 0, 'k', '3', 'h', 'e', 'l', 'l', 'o',
	}
	if !bytes.Equal(node[:len(want)], want) {
		t.Fatalf("encoded bytes = %x, want %x", node[:len(want)], want)
	}
	assertEntries(t, node, entries)
	assertEntries(t, newNode(nodeLeaf), nil)
}

func TestNodeAppendRange(t *testing.T) {
	entries := []testEntry{
		{key: []byte("a"), child: 0x0102030405060708},
		{key: []byte("m"), child: 22},
		{key: []byte("z"), child: 33},
	}
	src := testNode(nodeInternal, entries...)
	before := bytes.Clone(src)
	dst := newNode(nodeInternal)
	dst.setHeader(nodeInternal, 2)
	nodeAppendRange(dst, src, 0, 1, 2)
	assertEntries(t, dst, entries[1:])
	if !bytes.Equal(src[4:12], []byte{8, 7, 6, 5, 4, 3, 2, 1}) {
		t.Fatal("child page number is not encoded in little-endian order")
	}
	if !bytes.Equal(src, before) {
		t.Fatal("range copy changed the source")
	}
}

func TestNodeSplit(t *testing.T) {
	var twoLeaves, twoInternal []testEntry
	for i := 0; i < 6; i++ {
		twoLeaves = append(twoLeaves, testEntry{
			key: []byte(fmt.Sprintf("k%d", i)), value: bytes.Repeat([]byte{'v'}, 1000),
		})
		twoInternal = append(twoInternal, testEntry{
			key: bytes.Repeat([]byte{byte('a' + i)}, maxKeySize), child: uint64(i + 1),
		})
	}
	cases := []struct {
		name    string
		ntype   nodeType
		entries []testEntry
		count   uint16
	}{
		{"fits", nodeLeaf, []testEntry{{key: []byte("a"), value: []byte("v")}}, 1},
		{"two leaves", nodeLeaf, twoLeaves, 2},
		{"two internal nodes", nodeInternal, twoInternal, 2},
		{"large middle record", nodeLeaf, []testEntry{
			{key: bytes.Repeat([]byte{'a'}, 100), value: bytes.Repeat([]byte{'v'}, 100)},
			{key: bytes.Repeat([]byte{'m'}, maxKeySize), value: bytes.Repeat([]byte{'v'}, maxValueSize)},
			{key: bytes.Repeat([]byte{'z'}, 100), value: bytes.Repeat([]byte{'v'}, 100)},
		}, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := testNode(tc.ntype, tc.entries...)
			before := bytes.Clone(src)
			count, nodes := nodeSplit3(src)
			if count != tc.count {
				t.Fatalf("split count = %d, want %d", count, tc.count)
			}
			consumed := 0
			for _, node := range nodes[:count] {
				if node.kind() != tc.ntype || node.size() > PageSize || len(node) != PageSize {
					t.Fatalf("invalid split: type=%v, size=%d, buffer=%d", node.kind(), node.size(), len(node))
				}
				end := consumed + int(node.numKeys())
				assertEntries(t, node, tc.entries[consumed:end])
				consumed = end
			}
			if consumed != len(tc.entries) || !bytes.Equal(src, before) {
				t.Fatal("splitting lost entries or changed the source")
			}
		})
	}
}

func TestAppendKVOverflow(t *testing.T) {
	first := testEntry{key: bytes.Repeat([]byte{'a'}, maxKeySize), value: bytes.Repeat([]byte{'v'}, maxValueSize)}
	src := newNode(nodeLeaf).appendKV(first.key, first.value)
	before := bytes.Clone(src)
	second := testEntry{key: []byte("z"), value: bytes.Repeat([]byte{'w'}, 100)}
	dst := src.appendKV(second.key, second.value)
	assertEntries(t, dst, []testEntry{first, second})
	if dst.size() <= PageSize || len(dst) != 2*PageSize {
		t.Fatal("append must retain an oversized node for splitting")
	}
	if !bytes.Equal(src, before) {
		t.Fatal("append changed the source")
	}
	count, _ := nodeSplit3(dst)
	if count != 2 {
		t.Fatalf("split count = %d, want 2", count)
	}
}
