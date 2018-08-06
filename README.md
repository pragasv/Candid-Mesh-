#Candid Mesh implementation and face swap

This is an implementation of the candid mesh. We also extract 113 facefeature points. 
As well as the face swap function very similar to the one used by snapchat. 

This code needs a bit of time to run, in the sense , real time attachment of a webcam is not quite 
posible as an image needs 0.2 minutes to do the computations in a 4GB - 2.2 GHZ PC. 

The jocobian transformation for the dlib feature points creates the 113 face points. 

The Gaussian-Newton approximation is done to match the face colors after the face swap.

The ability to match the face swap with taking the facial emotions is an added advantage of this 
implementation. in the sence , if the mouth is open , the faceswap has the ability to match and
swap accordingly. 


![test_1](https://user-images.githubusercontent.com/22176286/43703915-d1a3cfc0-997b-11e8-8473-eecd40fe7b21.PNG)
**Marking of all 113 feature points**

![test_2](https://user-images.githubusercontent.com/22176286/43703954-ed70c226-997b-11e8-8220-39d191d98ec1.PNG)
**Marking of the candid mesh to all available faces**

![test_1](https://user-images.githubusercontent.com/22176286/43703981-feb72a3e-997b-11e8-9b00-7afa596c0eb6.PNG)
**Image after facewap**

