def get_semantic_analysis(df_error, hf_dataset):
    INDEX_KEY_MAPPING ={
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
    }

    for row in df_error.head(5):
        index = row["Index"]
        true_label = row["True Label"]
        predicted_label = row["Predicted Label"]

        print(f"Predicted: {INDEX_KEY_MAPPING.get(predicted_label)}")
        print(f"Ground Truth: {INDEX_KEY_MAPPING.get(true_label)}")
        print(f"Q: {hf_dataset[index]['question']}")
        print("\n".join(f"{INDEX_KEY_MAPPING.get(i)}: {a}" for i, a in enumerate(hf_dataset[index]["choices"]["text"])))


def rnn_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    questions, labels = zip(*batch)

    lengths = torch.tensor([q.shape[1] for q in questions])
    max_len = lengths.max()
    padded_questions = torch.stack(
        [torch.nn.functional.pad(q, (0, 0, 0, max_len - l)) for q, l in zip(questions, lengths)])

    labels = torch.stack(labels)

    return padded_questions, lengths, labels


def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch_size, num_choices, seq_len, emb_dim = x.shape

    lengths, perm_idx = lengths.sort(descending=True)
    x = x[perm_idx]
    x = x.view(batch_size * num_choices, seq_len, emb_dim)
    lengths = lengths.to("cpu").type(torch.int64)
    packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.repeat_interleave(num_choices), batch_first=True)

    h0 = torch.zeros(self.num_gru_layers * self.num_directions, batch_size * num_choices, self.gru.hidden_size).to(
        x.device)

    _, hn = self.gru(packed_x, h0)
    fnn_in = hn[-self.num_directions:, :, :].transpose(0, 1).flatten(-2, -1).view(batch_size, -1)
    fnn_in = self.norm(fnn_in)
    out = super().forward(fnn_in)
    perm_out = torch.empty_like(out)
    perm_out[perm_idx] = out
    return perm_out