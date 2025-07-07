import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(AlignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_feat, text_feat, relevant_clips_mask):
        """
        Calculates the alignment loss between audio and text features.

        Args:
            audio_feat (torch.Tensor): Audio features of shape (batch_size, num_clips, feat_dim).
            text_feat (torch.Tensor): Text features of shape (batch_size, num_words, feat_dim).
            relevant_clips_mask (torch.Tensor): A binary mask of shape (batch_size, num_clips)
                                                indicating relevant clips (1 for relevant, 0 for not).
        """
        # L2-normalize features
        audio_feat = F.normalize(audio_feat, p=2, dim=2)
        text_feat = F.normalize(text_feat, p=2, dim=2)

        # --- Local Regulator Loss ---
        # Cosine similarity between each audio clip and each text word
        # (batch_size, num_clips, num_words)
        s_loc = torch.bmm(audio_feat, text_feat.transpose(1, 2))

        # Mean pooling over text words to get similarity between each clip and global text
        # (batch_size, num_clips)
        s_loc_hat = torch.mean(s_loc, dim=2)

        # Sigmoid to get probabilities
        s_loc_hat = torch.sigmoid(s_loc_hat)

        # Local regularizer loss (binary cross-entropy)
        loss_local = F.binary_cross_entropy(s_loc_hat, relevant_clips_mask.float(), reduction='mean')

        # --- Global Regulator Loss ---
        # Global features by averaging
        g_audio = torch.mean(audio_feat, dim=1)
        g_text = torch.mean(text_feat, dim=1)

        # L2-normalize global features
        g_audio = F.normalize(g_audio, p=2, dim=1)
        g_text = F.normalize(g_text, p=2, dim=1)

        # Cosine similarity for contrastive loss
        logits = torch.mm(g_audio, g_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_global = F.cross_entropy(logits, labels)

        return loss_local, loss_global