import yaml



def get_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


def get_matcher():
    config = get_config()
    module_name = "Matching"
    feature_matcher = __import__(module_name, fromlist=[config["feature_matcher"]])
    feature_matcher = getattr(feature_matcher, config["feature_matcher"])
    return feature_matcher


def get_feature_detector():
    config = get_config()
    module_name = "Matching"
    feature_detector = __import__(module_name, fromlist=[config["feature_matcher"]])
    feature_detector = getattr(feature_detector, config["feature_matcher"])
    return feature_detector
