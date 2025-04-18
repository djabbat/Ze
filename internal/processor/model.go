package processor


type Counter struct {
    ID    uint32
    Data  [4]byte
    Value uint32
}

func (c *Counter) Increment(by uint32) {
    c.Value += by
}

func (c *Counter) Reset() {
    c.Value /= 2
}