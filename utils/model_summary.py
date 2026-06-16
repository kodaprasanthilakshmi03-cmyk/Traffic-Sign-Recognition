from keras.models import load_model
from io import StringIO
import sys
def get_model_summary():
    model = load_model("model/model.h5")  
    stream = StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    return stream.getvalue()
