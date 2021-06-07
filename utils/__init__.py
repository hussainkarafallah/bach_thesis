
def get_trainer(config):
    import recbole.utils, statics
    model_type = config['MODEL_TYPE']
    model_name = config['model']
    print(model_name)
    if model_name in statics.trainers:
        return statics.trainers[model_name]

    return recbole.utils.get_trainer(model_type , model_name)