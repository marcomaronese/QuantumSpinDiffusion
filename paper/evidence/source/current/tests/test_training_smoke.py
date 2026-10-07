import spin_diffusion_toy as toy


def test_reverse_training_reduces_loss_and_improves_on_prior():
    j = 1.0
    points = toy.make_direction_dataset(n_samples=120, sigma=0.22, seed=5)
    rho_data = toy.empirical_density(points, j)
    Jx, Jy, Jz, identity = toy.spin_operators(j)
    forward = toy.build_forward_trajectory(
        rho_data, Jx, Jy, Jz, D=1.0, dt=0.35, T=2
    )
    generators = toy.build_reverse_generators(Jx, Jy, Jz, identity)
    parameters, history = toy.train_reverse(
        forward,
        generators,
        layers=1,
        epochs=30,
        lr=0.05,
        seed=5,
    )

    prior = identity / identity.shape[0]
    generated = toy.generate_density(prior, parameters, generators)

    assert history[-1] < history[0]
    assert toy.frobenius_loss(generated, rho_data) < toy.frobenius_loss(
        prior, rho_data
    )
    toy.assert_physical_density(generated, "generated smoke-test state")
