class Memory:
    def __init__(self):
        self.history = []

    def store(self, data: dict):
        self.history.append(data)

    def get_all(self):
        return self.history