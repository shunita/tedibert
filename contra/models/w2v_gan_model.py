from itertools import chain
import torch
from contra.models.w2v_on_years import PretrainedOldNewW2V, read_w2v_model
from contra.utils.text_utils import TextUtils
from contra.models.model import FairEmbedding, Autoencoder


class FairEmbeddingW2V(FairEmbedding):
    def __init__(self, hparams):
        super(FairEmbeddingW2V, self).__init__(hparams)
        self.tokenizer = TextUtils()
        read_w2v_params = {'abstract_weighting_mode': hparams.abstract_weighting_mode,
                           'pubmed_version': hparams.pubmed_version,
                           'only_aact_data': hparams.only_aact_data}
        old_w2v = read_w2v_model(hparams.first_start_year, hparams.first_end_year, **read_w2v_params)
        new_w2v = read_w2v_model(hparams.second_start_year, hparams.second_end_year, **read_w2v_params)
        self.w2v = PretrainedOldNewW2V(old_w2v, new_w2v)
        self.autoencoder = Autoencoder(self.initial_embedding_size, self.embedding_size)

    def forward(self, batch):
        text = batch['text']
        # Tokenization is the same for old and new (we use word_tokenize for both).
        # embedding is done by a different model for "old" or "new" samples.
        tokenized_texts = [self.tokenizer.word_tokenize_abstract(t) for t in text]
        is_new = batch['is_new']
        sentence_embedding = self.w2v.embed_batch(tokenized_texts, is_new, self.device)

        reconstructed, latent = self.autoencoder(sentence_embedding)
        g_loss = self.L1Loss(sentence_embedding, reconstructed)
        return latent, g_loss

    def configure_optimizers(self):
        opt1_params = self.autoencoder.parameters()
        if self.do_ratio_prediction:
            opt1_params = chain(opt1_params, self.ratio_reconstruction.parameters())
        optimizer_1 = torch.optim.Adam(opt1_params, lr=self.hparams.learning_rate)
        optimizer_2 = torch.optim.Adam(self.discriminator.parameters(), 
                                       lr=self.hparams.learning_rate,
                                       weight_decay=self.hparams.regularize)
        return [optimizer_1, optimizer_2]