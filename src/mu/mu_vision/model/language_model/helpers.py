import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from .modeling_mu import CrossAttention


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


class VisCrossEncoderLayer(nn.Module):
    """
    VisCrossEncoderLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    Similar to the FlamingoLayer
    """

    def __init__(self, gated_cross_attn_layer, encoder_layer, gradient_checkpointing=False):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.encoder_layer = encoder_layer
        self.vis_x = None
        self.vis_attn_mask = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = gradient_checkpointing
        self.encoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x, vis_attn_mask=None):
        self.vis_x = vis_x
        self.vis_attn_mask = vis_attn_mask

    def forward(
        self,
        lang_x,
        attention_mask=None,
        encoder_mask=None,
        **encoder_layer_kwargs,
    ):
        is_mu_v2 = False
        if encoder_mask is not None:
            is_mu_v2 = True
        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")
            lang_x = self.gated_cross_attn_layer(lang_x, self.vis_x, encoder_attention_mask=self.vis_attn_mask)
        # Normal encoder layer
        if is_mu_v2 is False:
            lang_x = self.encoder_layer(lang_x, attention_mask=attention_mask)
        else:
            lang_x = self.encoder_layer(lang_x, encoder_mask=attention_mask)
        return lang_x


class VisCrossDecoderLayer(nn.Module):
    """
    VisCrossDecoderLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    Similar to the FlamingoLayer
    """

    def __init__(self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.vis_attn_mask = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = gradient_checkpointing
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x, vis_attn_mask=None):
        self.vis_x = vis_x
        self.vis_attn_mask = vis_attn_mask

    def forward(
        self,
        lang_x,
        enc_out,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        encoder_mask=None,
        decoder_mask=None,
        **decoder_layer_kwargs,
    ):
        is_mu_v2 = False
        if encoder_mask is not None:
            is_mu_v2 = True

        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            lang_x = self.gated_cross_attn_layer(lang_x, self.vis_x, encoder_attention_mask=self.vis_attn_mask)
            # print("Go through the gated_cross_attn_layer")

        # Normal decoder layer
        if is_mu_v2 is False:
            lang_x = self.decoder_layer(
                lang_x,
                enc_out=enc_out,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
        else:
            lang_x = self.decoder_layer(lang_x, enc_out=enc_out, encoder_mask=encoder_mask, decoder_mask=decoder_mask)
        return lang_x


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.config = PretrainedConfig()
        self.config.n_embd = dim
        self.config.n_head = heads
        self.config.n_kv_heads = self.config.n_head
        self.attn = CrossAttention(self.config)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, media, encoder_attention_mask=None):
        x = self.attn(x, enc_out=media, encoder_attention_mask=encoder_attention_mask) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x


class VisionCrossAttnMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing=False,
        use_vision_cross_which_tower="encoder",
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(dim=lang_hidden_size, dim_visual=vis_hidden_size)
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self.init_flamingo_layers(gradient_checkpointing, use_vision_cross_which_tower=use_vision_cross_which_tower)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_vision_x = False

    def init_flamingo_layers(self, gradient_checkpointing, use_vision_cross_which_tower="encoder"):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        if use_vision_cross_which_tower == "encoder":
            vis_layer_class = VisCrossEncoderLayer
            print("use vision cross attn for encoder tower")
        elif use_vision_cross_which_tower == "decoder":
            vis_layer_class = VisCrossDecoderLayer
            print("use vision cross attn for decoder tower")
        else:
            raise ValueError("cannot apply vision cross attn to tower {}".format(use_vision_cross_which_tower))
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    vis_layer_class(gated_cross_attn_layer, decoder_layer, gradient_checkpointing)
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )
        del self.old_decoder_blocks
        del self.gated_cross_attn_layers
