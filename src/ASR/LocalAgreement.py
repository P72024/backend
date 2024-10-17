class LocalAgreement:
    confirmed_text: str = ""
    unconfirmed_text: str = ""

    def longest_common_sequence(self, unconfirmed_incoming_text, unconfirmed_text):
        words1 = unconfirmed_text.split()
        words2 = unconfirmed_incoming_text.split()

        longest_seq = []
        current_seq = []

        for i in range(len(words1)):
            if i < len(words2) and words1[i] == words2[i]:
                current_seq.append(words1[i])
            else:
                if len(current_seq) > len(longest_seq):
                    longest_seq = current_seq
                current_seq = []

        # Handle case where the longest sequence ends at the last word
        if len(current_seq) > len(longest_seq):
            longest_seq = current_seq

        return " ".join(longest_seq)

    def confirm_tokens(self, incoming_text: str) -> str:
        # Find the unconfirmed part of the incoming text (ignoring the previously confirmed part)
        unconfirmed_incoming_text = incoming_text[len(self.confirmed_text):].strip()

        # Find the longest common sequence between the new unconfirmed incoming text and the unconfirmed text
        lcs = self.longest_common_sequence(unconfirmed_incoming_text, self.unconfirmed_text)

        # Update the confirmed text by appending the longest common sequence found
        if lcs:
            if self.confirmed_text:
                self.confirmed_text += " " + lcs
            else:
                self.confirmed_text = lcs

        # Store the remaining unconfirmed part of the incoming text for the next input
        self.unconfirmed_text = unconfirmed_incoming_text[len(lcs):].strip()

        return self.confirmed_text