Mosse Léo
Sontag Thomas
Grebert Pierre
3A Apprentissage

Nous avons tous travaillé équitablement malgré l'unique commit sur git ! (Travail partagé via discord)

Nous avons téléchargé le dataset OxfordIIITPet. Il existe un exemple des images du dataset dans le code (en cmap="gray" car je n'ai pas réussi à les avoir en vrai couleur):

Le dataset contient des images RGB de tailles variables réparties en 37 classes. Nous avons donc dû resizer les images dans le transform (en 250x250, bon compromis temps d'entraînement-qualité de résultats), et adapter le modèle ViT pour qu'il utilise les 3 channels de couleur. Nous avons également adapté les batch size et les patch size pour réguler la consommation de ram de l'application.

N'hésitez pas à copier le fichier .ipynb et à le run dans un google collab ou dans l'environnement de votre choix :)

Nombre d'images d'entraînement: 3680
Nombre d'image de test: 3669

La fonction train_model permets d'entraîner le modèle sur le dataset d'entraînement, la fonction eval_model permets d'évaluer le modèle entraîné précedemment sur le dataset d'évaluation.

Voici les résultats d'un entraînement sur 25 epochs en 26 minutes 33 s (GTX 1650):
----------

Epoch: 0

Loss: 1595.0327961444855

Accuracy: 6.2771739130434785 % (231 / 3680)

----------
Epoch: 1

Loss: 1549.3069727420807

Accuracy: 8.26086956521739 % (304 / 3680)

----------

Epoch: 2

Loss: 1503.7383060455322

Accuracy: 10.081521739130435 % (371 / 3680)

----------

Epoch: 3

Loss: 1462.467333316803

Accuracy: 12.581521739130435 % (463 / 3680)

----------

Epoch: 4

Loss: 1418.9557387828827

Accuracy: 14.918478260869565 % (549 / 3680)

----------

Epoch: 5

Loss: 1367.44597530365

Accuracy: 16.956521739130434 % (624 / 3680)

----------

Epoch: 6

Loss: 1315.783386349678

Accuracy: 20.434782608695652 % (752 / 3680)

----------

Epoch: 7

Loss: 1250.1559344530106

Accuracy: 23.505434782608695 % (865 / 3680)

----------

Epoch: 8

Loss: 1174.4544525146484

Accuracy: 26.956521739130434 % (992 / 3680)

----------

Epoch: 9

Loss: 1091.810905456543

Accuracy: 32.608695652173914 % (1200 / 3680)

----------

Epoch: 10

Loss: 1003.6593238115311

Accuracy: 37.22826086956522 % (1370 / 3680)

----------

Epoch: 11

Loss: 918.1153181791306

Accuracy: 43.77717391304348 % (1611 / 3680)

----------

Epoch: 12

Loss: 829.8904433250427

Accuracy: 48.39673913043478 % (1781 / 3680)

----------

Epoch: 13

Loss: 733.0716485381126

Accuracy: 54.40217391304348 % (2002 / 3680)

----------

Epoch: 14

Loss: 654.5808556973934

Accuracy: 59.29347826086956 % (2182 / 3680)

----------

Epoch: 15

Loss: 559.1795224547386

Accuracy: 67.52717391304348 % (2485 / 3680)

----------

Epoch: 16

Loss: 476.02283251285553

Accuracy: 71.46739130434783 % (2630 / 3680)

----------

Epoch: 17

Loss: 386.29163524508476

Accuracy: 78.55978260869566 % (2891 / 3680)

----------

Epoch: 18

Loss: 335.6575016230345

Accuracy: 81.08695652173913 % (2984 / 3680)

----------

Epoch: 19

Loss: 278.35613030195236

Accuracy: 84.97282608695652 % (3127 / 3680)

----------

Epoch: 20

Loss: 217.13183546811342

Accuracy: 88.72282608695652 % (3265 / 3680)

----------

Epoch: 21

Loss: 190.2041505649686

Accuracy: 89.86413043478261 % (3307 / 3680)

----------

Epoch: 22

Loss: 149.77935491874814

Accuracy: 92.36413043478261 % (3399 / 3680)

----------

Epoch: 23

Loss: 128.06930090487003

Accuracy: 93.55978260869566 % (3443 / 3680)

----------

Epoch: 24

Loss: 109.98742446862161

Accuracy: 94.6195652173913 % (3482 / 3680)


Cependant, on remarque que le modèle a overfitté après ces 25 epochs car les résultats de sont pas du tout concluants : 

Evaluation accuracy: 10.616812227074236 % (389 / 3664)

Après avoir réalisé d'autre tests, en 11 epoch, le modèle faisait 11.5% de bonne réponse, alors qu'après 12 époch, il était de retour à 10.6. On voit donc un vrai problème d'overfitting.

Pour corriger ce problème d'overfitting, nous avons décidé d'ajouter artificiellement des images au dataset en ajoutant des altérations aléatoires des images(rotations verticales et/ou horizontales)

L'appel à len(train_dataset) retourne toujours le même nombre d'images car cela vous affiche le nombre d'images dans le dossier téléchargé. Cependant, lorsqu'on itère sur le dataset (avec le dataloader), cela va appliquer ces augmentations aléatoires. Admettons un exemple simple avec une seule augmentation de type symétrie horizontale (https://pytorch.org/vision/0.15/auto_examples/plot_transforms.html#randomhorizontalflip) avec une probabilité de 50%, l'ajout de cette augmentation vous double le nombre d'images du dataset (3680 images de base + 3680 images flip).

Après 10 epoch, on atteint uniquement une précision de 10% environ, contrairement aux 40% précédents :

----------

Epoch: 0

Loss: 1709.229103088379

Accuracy: 3.2880434782608696 % (121 / 3680)

----------

Epoch: 1

Loss: 1680.4841358661652

Accuracy: 3.233695652173913 % (119 / 3680)

----------

Epoch: 2

Loss: 1639.8549852371216

Accuracy: 4.565217391304348 % (168 / 3680)

----------

Epoch: 3

Loss: 1604.1130244731903

Accuracy: 6.2228260869565215 % (229 / 3680)

----------

Epoch: 4

Loss: 1581.4872334003448

Accuracy: 6.413043478260869 % (236 / 3680)

----------

Epoch: 5

Loss: 1562.5292735099792

Accuracy: 7.663043478260869 % (282 / 3680)

----------

Epoch: 6

Loss: 1546.164597272873

Accuracy: 8.505434782608695 % (313 / 3680)

----------

Epoch: 7

Loss: 1531.7870779037476

Accuracy: 9.184782608695652 % (338 / 3680)

----------

Epoch: 8

Loss: 1521.6575982570648

Accuracy: 9.728260869565217 % (358 / 3680)

----------

Epoch: 9

Loss: 1501.5961372852325

Accuracy: 10.353260869565217 % (381 / 3680)


Cependant, cela semble se traduire en un apprentissage plus profond et plus proche des résultats d'entraînement :

Evaluation accuracy: 8.324235807860262 % (305 / 3664)

Si l'on effectue 10 nouvelles epoch sur le même modèle, nous obtenons les résultats suivants :

----------

Epoch: 0

Loss: 1487.6564829349518

Accuracy: 11.08695652173913 % (408 / 3680)

----------

Epoch: 1

Loss: 1454.0886657238007

Accuracy: 12.581521739130435 % (463 / 3680)

----------

Epoch: 2

Loss: 1433.3625659942627

Accuracy: 13.994565217391305 % (515 / 3680)

----------

Epoch: 3

Loss: 1410.3271868228912

Accuracy: 15.16304347826087 % (558 / 3680)

----------

Epoch: 4

Loss: 1388.8576029539108

Accuracy: 16.005434782608695 % (589 / 3680)

----------

Epoch: 5

Loss: 1367.7089103460312

Accuracy: 16.304347826086957 % (600 / 3680)

----------

Epoch: 6

Loss: 1342.4329944849014

Accuracy: 18.39673913043478 % (677 / 3680)

----------

Epoch: 7

Loss: 1306.2374782562256

Accuracy: 19.782608695652176 % (728 / 3680)

----------

Epoch: 8

Loss: 1265.1505917310715

Accuracy: 22.96195652173913 % (845 / 3680)

----------

Epoch: 9

Loss: 1234.937785744667

Accuracy: 23.695652173913043 % (872 / 3680)

Et à l'évaluation ... 

Evaluation accuracy: 9.252183406113538 % (339 / 3664)

Ce qui est à peine 1% mieux que 10 epochs auparavant.

Après avoir échangé avec le professeur, nous avons décidé d'essayer d'augmenter le nombre d'images, mais nous sommes restés en échec, le code que nous avons essayé est dans le fichier mais il est commenté car il ne sert pas, dans le code commenté il y a l'explication.