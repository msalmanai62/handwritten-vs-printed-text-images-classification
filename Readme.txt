How to use?
Create virtual enviroment and activate it. Then go to project folder and open cmd. Then install all the libraries using this command
pip install -r requirements.txt

After environment setup completetion run this command to interact with the streamlit app
streamlit run app.py


Some Notes:
I have removed image contouring part because there was too much bad and annoying results using this. Instead model takes single image and predicts whether it contains humanwritten or printed text. So, use those images that contains only one text(either human written or printed).

Please not that however CNN custom model and transfer learning model have good accuracy but they can make mistakes most of time. Reason is that our model were trained on dataset that contains too much small images containing almost single word or line of words. So making predictions on larger images may result in false predictions.

Later I have added some images data from myside to train transformer model. So it achieved good accuracy as well as making accurate predictions.

I have created Streamlit GUI becuase your provided GUI code was too simple.

If you want to detect both human written and printed text in sigle image, it is an object detection problem and it requires data labeling which is time consuming task. But No doubt, the better approch is using object detection mechanism. But it was out of this project's scope.


for queries
 zero 325 saat saat doo chaar aik aik doo