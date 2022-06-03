from handler import SentenceTransformerHandler
_service = SentenceTransformerHandler()
def handle(data, context):
    """
    Entry point for SentenceTransformerHandler handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)
            print('ENTERING INITIALIZATION')
        if data is None:
            return None
        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)
        return data
    except Exception as e:
        raise Exception("Unable to process input data. " + str(e))