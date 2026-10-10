"""SDF scale initialization of the existing decoder, without an extra field term.

Softplus(beta=10) has a substantial positive offset. Center/radius calibration
below only sets initial weights/biases; it is not a loss, template or observation.
"""
import math
import torch


@torch.no_grad()
def initialize_decoder(model, radius=0.3):
    d=model.decoder
    width=d.a[0].out_features
    if width%2:
        raise ValueError('Paired direction initialization needs an even width')
    device=d.b[-1].weight.device
    index=torch.arange(width//2,device=device,dtype=torch.float32)
    zz=1-2*(index+.5)/(width//2)
    angle=index*(math.pi*(3-math.sqrt(5)))
    rr=(1-zz.square()).sqrt()
    directions=torch.stack([rr*angle.cos(),rr*angle.sin(),zz],-1)
    directions=torch.cat([directions,-directions])
    # Paired opposite directions eliminate a random linear tilt at startup.
    # All these weights remain free parameters during ordinary main training.
    for layer in [d.a[0],d.a[2],d.b[0],d.b[2]]:
        layer.weight.zero_();layer.bias.zero_()
    d.a[0].weight[:,:3].copy_(5*directions)
    for layer in [d.a[2],d.b[0],d.b[2]]:
        layer.weight[:,:width].copy_(torch.eye(width,device=device))
    torch.nn.init.normal_(d.a[0].weight[:,3:],0,1e-4)
    torch.nn.init.normal_(d.b[0].weight[:,width+3:],0,1e-4)
    d.b[-1].weight.fill_(1/width);d.b[-1].bias.zero_()
    q=torch.cat([torch.zeros(1,3,device=device),radius*directions])
    latent=torch.zeros(len(q),model.encoder.pool[-1].out_features,device=device)
    value=d(q,latent,0)
    center=value[0].clone()
    slope=(value[1:].mean()-center)/radius
    if not torch.isfinite(slope) or slope<=0:
        raise ValueError('Invalid geometric initialization scale')
    d.b[-1].weight.div_(slope)
    d.b[-1].bias.fill_(-center/slope-radius)
    return dict(radius_normalized=radius,softplus_scale_correction=float(slope),
                center_field=float(d(q[:1],latent[:1],0)[0]),
                mean_radius_field=float(d(q[1:],latent[1:],0).mean()))
