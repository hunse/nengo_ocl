from base import Model
from base import Simulator

def test_bp(Sim=Simulator):
    m = Model()

    i_inputs = range(n_inputs)
    i_hiddens = range(n_hiddens)
    i_outputs = range(n_outputs)
    i_targets = range(n_outputs)

    x_in = [m.signal() for i in i_inputs]
    p_in = [m.population(bias=None) for i in i_inputs]
    p_hid = [m.population(bias=None) for i in i_hiddens]
    p_out = [m.population(bias=None) for i in i_outputs]
    x_out = [m.signal() for i in i_outputs]
    x_targ = [m.signal() for i in i_outputs]
    x_grad = [m.signal() for i in i_outputs]

    for ii in i_inputs:
        m.encoder(x_in[ii], p_in[ii], weights=None)

    for ii in i_inputs:
        for jj in i_hiddens:
            m.connect(p_in[ii], p_hid[jj], weights=None)

    for jj in i_hiddens:
        for kk in i_outputs:
            m.connect(p_hid[jj], p_out[kk], weights=None)

    for kk in i_outputs:
        m.decoder(p_out[kk], x_out[kk], weights=None)

    for kk in i_outputs:
        m.transform( 1.0, x_targ[kk], x_grad[kk])
        m.transform(-1.0, x_out[kk], x_grad[kk])

