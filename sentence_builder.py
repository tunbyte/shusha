class SentenceBuilder:
    def __init__(self, stable_frames=15, cooldown_frames=20):
        self.stable_frames = stable_frames
        self.cooldown_frames = cooldown_frames

        self.current_letter = None
        self.frame_count = 0
        self.cooldown = 0

        self.current_word = ""
        self.sentence = []


    def update(self, letter):
        if self.cooldown > 0:
            self.cooldown -= 1
            return False

        if letter is None:
            self.current_letter = None
            self.frame_count = 0
            return False

        if letter == "SPACE":
            if self.current_word:
                self.sentence.append(self.current_word)
                self.current_word = ""
                self.current_letter = None
                self.frame_count = 0
                self.cooldown = self.cooldown_frames
                return True
            return False

        if letter == "DEL":
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.current_letter = None
                self.frame_count = 0
                self.cooldown = self.cooldown_frames
                return True
            return False

        if letter == self.current_letter:
            self.frame_count += 1
        else:
            self.current_letter = letter
            self.frame_count = 1

        if self.frame_count >= self.stable_frames:
            self.current_word += letter
            self.frame_count = 0
            self.cooldown = self.cooldown_frames
            return True
        return False

    def get_display_text(self):
        sentence_str = " ".join(self.sentence)
        if self.current_word:
            if sentence_str:
                return sentence_str + " " + self.current_word
            return self.current_word
        return sentence_str

    def get_progress(self):
        if self.current_letter is None:
            return 0, self.stable_frames
        return self.frame_count, self.stable_frames

    def clear(self):
        self.current_letter = None
        self.frame_count    = 0
        self.cooldown       = 0
        self.current_word   = ""
        self.sentence       = []