class Configuration:
    def __init__(self):
        self.src_vocab_size = 58731   #24995-java #15552-python #ourdata-58731
        self.target_vocab_size = 48046   #11649-java #3442-python #ourdata-48046
        self.emb_dim = 64
        self.hid_dim = 512
        self.n_layers = 1
        self.dropout = 0.1
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.max_length = 400  #150-java
        # 17060
        # 7682
