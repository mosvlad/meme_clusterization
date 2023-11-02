# MEME CLUSTERIZATION

Example project for calculate image embeddings([ResNet-152](https://pytorch.org/hub/pytorch_vision_resnet/)) and use dimension reduce([t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)) for clusterization([DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)) and plotting. 


![image](https://github.com/mosvlad/meme_clusterization/assets/31764930/e239dc2f-0a85-4202-901a-52bb4f955308)


---

* Calculate image embeddings for every image in memes folder;
* Reduce dimension of embeddings to 2D;
* Use clustering algorithm for find clusters of memes;
* Plot results ( _plotting takes some time_ )

---
## Prepare data
For calc embeddings use ([calc_embeddings.py](https://github.com/mosvlad/meme_clusterization/blob/master/calc_embeddings.py)).

Change path for your images folder [here](https://github.com/mosvlad/meme_clusterization/blob/00f6abb392895f71d1810bbc3e42b58fac4ea06f/calc_embeddings.py#L9C1-L9C66)

---
## Run clustering

```
git clone https://github.com/mosvlad/meme_clusterization
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 main.py

```

For run clustering you need precalculated embeddings. For this project embedding store as __PYTHON DICT__, where:

* __key__ - __path to image__;
* __value__ - __1024 float vector__;

---
## Example of clustering MEMES

![image](https://github.com/mosvlad/meme_clusterization/assets/31764930/771cd032-282a-47e6-8eb2-f675b0806027)
![image](https://github.com/mosvlad/meme_clusterization/assets/31764930/2c62ea7e-7e07-4b24-95d9-bb1317761d8b)
![image](https://github.com/mosvlad/meme_clusterization/assets/31764930/cd1f292b-7679-48e0-b7cc-b1759e0e2250)



