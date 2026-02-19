class LRUCache {
    int cap;
    Map<Integer, Node> cache;
    Node front;
    Node end;

    class Node {
        int value;
        int key;
        Node prev;
        Node next;

        public Node(int key, int val) {
            this.value = val;
            this.key = key;
        }
    }

    public LRUCache(int capacity) {
        cap = capacity;
        cache = new HashMap<>();
        front = new Node(0,0);
        end = new Node(0,0);
        front.next = end;
        end.prev = front;
    }
    
    public int get(int key) {
        if (this.cache.containsKey(key)) {
            Node ans = this.cache.get(key);
            removeNode(ans);
            addNode(ans);
            return ans.value;
        } else {
            return -1;
        }
    }

    private void removeNode(Node node) {
        Node prev = node.prev;
        Node next = node.next;
        prev.next = next;
        next.prev = prev;
    }

    private void addNode(Node node) {
        Node cur_front = front.next;
        front.next = node;
        node.prev = front;
        cur_front.prev = node;
        node.next = cur_front;
    }
    
    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            Node cur = cache.get(key);
            cur.value = value;
            removeNode(cur);
            addNode(cur);
        } else {
            if (cache.size() >= this.cap) {
                Node cur_last = end.prev;
                removeNode(cur_last);
                cache.remove(cur_last.key);
            }
            Node new_add = new Node(key, value);
            cache.put(key, new_add);
            addNode(new_add);
        }
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */