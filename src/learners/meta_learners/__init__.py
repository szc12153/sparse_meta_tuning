from .sparse_interpolated_experts import SparseInterpolatedExperts

def get_meta_learner(base_learner, args):
    AVALIABLE_MODELS = {
        "SPARSE_INTERPOLATED_EXPERTS": SparseInterpolatedExperts,
    }
    learner = AVALIABLE_MODELS.get(args.meta_learner.model)(base_learner,args)
    assert learner
    return learner


