import numpy as np
import pytest
import torch

from spin_quantum_diffusion.quantum.coherent_readout import coherent_directions_from_uniforms
from spin_quantum_diffusion.quantum.euclidean import RadialSpinChart


@pytest.mark.parametrize("kind",["stereographic","sinh_stereographic"])
@pytest.mark.parametrize("length",[.5,1.,2.])
def test_radial_maps_roundtrip_core_and_far_tails_without_truncation(kind,length):
    chart=RadialSpinChart(np.array([2.,-3.]),np.array([.7,1.5]),length,kind)
    raw=np.array([[0.,0.],[.1,-.2],[2.,3.],[30.,-20.]])*chart.scale+chart.location
    directions,clipped=chart.encode(raw)
    np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1,atol=1e-14)
    assert not clipped.any()
    np.testing.assert_allclose(chart.decode(directions),raw,atol=1e-11,rtol=1e-12)
    with pytest.raises(ValueError):
        chart.decode(np.array([[0.,0.,-1.]]))


def test_inverse_growth_matches_exact_uniform_tail_survivals():
    radius=np.array([.1,.5,1.,2.,5.])
    raw=np.c_[radius,np.zeros(len(radius))]
    for kind in ("stereographic","sinh_stereographic"):
        chart=RadialSpinChart(np.zeros(2),np.ones(2),1.,kind)
        directions,_=chart.encode(raw)
        survival=(1+directions[:,2])/2
        expected=1/(1+radius**2) if kind=="stereographic" else 1/np.cosh(radius)**2
        np.testing.assert_allclose(survival,expected,atol=1e-14)


def test_logarithmic_inverse_has_finite_known_radial_moments():
    z,weights=np.polynomial.legendre.leggauss(512)
    directions=np.c_[np.sqrt(1-z*z),np.zeros(len(z)),z]
    chart=RadialSpinChart(np.zeros(2),np.ones(2),1.,"sinh_stereographic")
    radius=np.linalg.norm(chart.decode(directions),axis=1)
    assert weights@radius/2==pytest.approx(1,abs=1e-5)
    assert weights@(radius**2)/2==pytest.approx(2*np.log(2),abs=1e-4)
    # Ordinary stereo second moments keep increasing with pole resolution.
    moments=[]
    for count in (64,256):
        zz,ww=np.polynomial.legendre.leggauss(count)
        moments.append(ww@((1-zz)/(1+zz))/2)
    assert moments[1]>moments[0]+2


def test_continuous_quantiles_have_exact_overlap_and_analytic_moments():
    count=65536
    uniform=(torch.arange(count,dtype=torch.float64)+.5)/count
    directions=torch.tensor([0.,0.,1.],dtype=torch.float64).expand(count,3)
    observed=coherent_directions_from_uniforms(directions,5,uniform,uniform.flip(0))
    torch.testing.assert_close(observed.norm(dim=-1),torch.ones(count,dtype=torch.float64),atol=1e-14,rtol=0)
    expected=2*uniform.pow(1/6)-1
    torch.testing.assert_close(observed[:,2],expected,atol=1e-14,rtol=0)
    assert float(observed[:,2].mean())==pytest.approx(5/7,abs=5e-6)
    assert float(observed[:,2].square().mean())==pytest.approx(4/7,abs=5e-6)


def test_local_angular_stretches_match_finite_differences():
    for kind in ("stereographic","sinh_stereographic"):
        chart=RadialSpinChart(np.zeros(2),np.ones(2),1.,kind)
        point=np.array([[.8,0.]])
        stretches=chart.local_stretches(point)[0]
        derivatives=[]
        for delta in (np.array([[1e-6,0]]),np.array([[0,1e-6]])):
            plus=chart.encode(point+delta)[0]
            minus=chart.encode(point-delta)[0]
            derivatives.append(np.linalg.norm(plus-minus)/(2e-6))
        np.testing.assert_allclose(stretches,derivatives,atol=1e-9)
