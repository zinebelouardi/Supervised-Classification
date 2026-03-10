{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/zinebelouardi/Classification-Supervis-e/blob/main/Classification_Supervis%C3%A9e_Benchmarking_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QUP-0fdHYcTH"
      },
      "outputs": [],
      "source": [
        "\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from tensorflow import keras\n",
        "import pandas as pd\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# CHARGEMENT ET PRÉPARATION DES DONNÉES"
      ],
      "metadata": {
        "id": "mqs053BrbmFz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Charger CIFAR-10\n",
        "(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()\n",
        "\n",
        "# Garder seulement Oiseau (classe 2) et Camion (classe 9)\n",
        "mask_train = (y_train == 2) | (y_train == 9)\n",
        "mask_test = (y_test == 2) | (y_test == 9)\n",
        "\n",
        "x_train = x_train[mask_train.flatten()]\n",
        "y_train = y_train[mask_train.flatten()]\n",
        "x_test = x_test[mask_test.flatten()]\n",
        "y_test = y_test[mask_test.flatten()]\n",
        "\n",
        "# Convertir: Oiseau=0, Camion=1\n",
        "y_train = (y_train == 9).astype(int)\n",
        "y_test = (y_test == 9).astype(int)\n",
        "\n",
        "print(f\" Données train: {len(x_train)} images\")\n",
        "print(f\" Données test: {len(x_test)} images\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BNm2RferZmzO",
        "outputId": "831eee91-7980-4702-ad0e-0c0cdbd90889"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n",
            "\u001b[1m170498071/170498071\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 0us/step\n",
            " Données train: 10000 images\n",
            " Données test: 2000 images\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        " # 2. EXTRACTION DES CARACTÉRISTIQUES\n"
      ],
      "metadata": {
        "id": "CvIvK505cEvS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def extraire_features(images):\n",
        "    \"\"\"Extraire les caractéristiques de couleur et texture\"\"\"\n",
        "    features = []\n",
        "\n",
        "    for img in images:\n",
        "        img = img / 255.0  # Normalisation\n",
        "\n",
        "        # 1. Histogramme couleur (24 features)\n",
        "        hist_r = np.histogram(img[:,:,0], bins=8, range=(0,1))[0]\n",
        "        hist_g = np.histogram(img[:,:,1], bins=8, range=(0,1))[0]\n",
        "        hist_b = np.histogram(img[:,:,2], bins=8, range=(0,1))[0]\n",
        "\n",
        "        # 2. Moments couleur (6 features)\n",
        "        mean_rgb = [img[:,:,i].mean() for i in range(3)]\n",
        "        std_rgb = [img[:,:,i].std() for i in range(3)]\n",
        "\n",
        "        # 3. Texture (4 features)\n",
        "        gray = img.mean(axis=2)\n",
        "        texture = [gray.mean(), gray.std(),\n",
        "                   np.sum(gray**2)/gray.size,\n",
        "                   -np.sum(gray * np.log(gray + 1e-10))/gray.size]\n",
        "\n",
        "        # Combiner toutes les features\n",
        "        f = np.concatenate([hist_r, hist_g, hist_b, mean_rgb, std_rgb, texture])\n",
        "        features.append(f)\n",
        "\n",
        "    return np.array(features)\n",
        "\n",
        "# Extraire les features\n",
        "X_train = extraire_features(x_train)\n",
        "X_test = extraire_features(x_test)\n",
        "\n",
        "print(f\" {X_train.shape[1]} caractéristiques extraites par image\")\n",
        "\n",
        "# Normaliser les features\n",
        "scaler = StandardScaler()\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L3Z8EUg9bJZT",
        "outputId": "ab5f8792-11ba-4525-a96f-2ff67ee05aa8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " 34 caractéristiques extraites par image\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#  ENTRAÎNEMENT DU MODÈLE SVM\n"
      ],
      "metadata": {
        "id": "-JtisxyOcc39"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "svm = SVC(kernel='rbf', C=1.0, random_state=42)\n",
        "svm.fit(X_train, y_train)\n",
        "print(\" Modèle SVM entraîné avec succès\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YHtDu8YrcZvX",
        "outputId": "3e43ddf2-7336-49c0-8b24-cb0193961e99"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " Modèle SVM entraîné avec succès\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# PRÉDICTION SUR DES IMAGES DE VALIDATION\n"
      ],
      "metadata": {
        "id": "JN-STVXpcodd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Prendre 1 oiseau et 1 camion\n",
        "idx_oiseau = np.where(y_test == 0)[0][0]\n",
        "idx_camion = np.where(y_test == 1)[0][5]\n",
        "\n",
        "# Prédire\n",
        "pred_oiseau = svm.predict([X_test[idx_oiseau]])[0]\n",
        "pred_camion = svm.predict([X_test[idx_camion]])[0]\n",
        "\n",
        "classes = ['Oiseau', 'Camion']\n",
        "\n",
        "# Récupérer les vraies classes (conversion en int)\n",
        "vrai_oiseau = int(y_test[idx_oiseau])\n",
        "vrai_camion = int(y_test[idx_camion])\n",
        "\n",
        "# Afficher les images et prédictions\n",
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
        "\n",
        "axes[0].imshow(x_test[idx_oiseau])\n",
        "axes[0].set_title(f'Vrai: {classes[vrai_oiseau]}\\nPrédit: {classes[pred_oiseau]}',\n",
        "                  fontsize=12, fontweight='bold')\n",
        "axes[0].axis('off')\n",
        "\n",
        "axes[1].imshow(x_test[idx_camion])\n",
        "axes[1].set_title(f'Vrai: {classes[vrai_camion]}\\nPrédit: {classes[pred_camion]}',\n",
        "                  fontsize=12, fontweight='bold')\n",
        "axes[1].axis('off')\n",
        "\n",
        "plt.suptitle('Prédictions du SVM', fontsize=14, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"Image 1 → Vrai: {classes[vrai_oiseau]}, Prédit: {classes[pred_oiseau]}\")\n",
        "print(f\"Image 2 → Vrai: {classes[vrai_camion]}, Prédit: {classes[pred_camion]}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 538
        },
        "id": "njzhX5HWct_G",
        "outputId": "5a10a197-525f-4925-b699-2e2064717b4a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipython-input-2993137555.py:12: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)\n",
            "  vrai_oiseau = int(y_test[idx_oiseau])\n",
            "/tmp/ipython-input-2993137555.py:13: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)\n",
            "  vrai_camion = int(y_test[idx_camion])\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x400 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAw8AAAGNCAYAAABXO00sAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAWVhJREFUeJzt3XmYFPW59/9P792zL8CwryKrouKOBuJRiKiocUFjBJfHxGiMmpMcTeITXGJijnqMiUmeGHdjRFEUDRE0CkpwQ0VFAyK77DDMMHuv9fuD39SZcWa4vxgU0Pfrurxkuu+u+lZVd1XdXd39CXie5wkAAAAADME9PQAAAAAA+waaBwAAAABOaB4AAAAAOKF5AAAAAOCE5gEAAACAE5oHAAAAAE5oHgAAAAA4oXkAAAAA4ITmAcBXxkMPPaRAIKBEIqHZs2fv6eEAALDPoXkA8JWwaNEife9731MikdCzzz6rcePG7dLjA4GA/98DDzzg3/7AAw+0uu/zNmbMGH9eF1xwwec+vz1p7ty5rdbtqlWr9vSQAOArj+YBwBfu0yeFLf8rKCjQ0KFDdcUVV2jFihW7ZX61tbU688wzJUl/+9vfdPzxx++W6e5uX6XGYG+1ePFiXXLJJRo4cKASiYTi8bh69Oihgw8+WOeff75uv/12pdNpSdLUqVNbPXcff/zxDqd7ww03tKp97733JLXe5s3/vfPOO+1O46ijjmpTS0MF4IsW3tMDAICW6uvrtXjxYi1evFj33XefZsyY8W+f7L/77rs699xzNXbsWB199NG7aaQ7HHbYYbr11lt36zR35nvf+55OPvlkSdLw4cO/sPl+FTz33HM67bTTlEqlWt2+fv16rV+/Xu+++67+8pe/6OKLL1ZJSYlOO+00lZSUqLq6WpL08MMP6+yzz2532n/5y1/8fx900EEaMWJEh+P47W9/2+rqliQtWLBAr7/++mdbMADYjWgeAOxxEydO1KGHHqpUKqXXXntNf/vb3yRJDQ0NOv/887Vq1SrFYjFzOjU1NSoqKmpz+7HHHqtjjz12t49bkoYNG6Zhw4Z9LtNuz8SJE7+weX2VZLNZ/Z//83/8xqG8vFxnn322evXqpYaGBi1ZskSvvPKKNm/e7D8mHo9r4sSJ+tOf/iRJmjVrlrZs2aLOnTu3mvarr76qZcuW+X9bV5WmTp2qW2+9tdV07rzzzn93EQFgt+BjSwD2uG984xv60Y9+pJ/+9Kd69tlndd555/n3bdy4UfPnz5fU9uNOy5Yt02233aYhQ4YoFotp0qRJ/uNyuZwefvhhjR07Vl26dFE0GlWnTp00btw4Pfnkk+2OI5PJ6JZbbtHAgQMVi8U0YMAA/eIXv/A/ptIe6zsPmUxG9913n8aOHauKigpFo1F17txZRx55pG644QZJ0vXXX69AIKCXX37Zf9yDDz7Y7sdTrI82LV26VN/73vc0aNAg5eXlKS8vT/vvv7+++93vasmSJW3qL7jgAn96Y8aM0YYNG/Sd73xH3bp1UywW05AhQ/TnP/+5zeO2bt2qH/3oRxo2bJjy8/MVjUbVtWtXHX744fr+97+/S++SV1ZW6tJLL1VFRYUSiYQOPfRQPfbYYzt9zM7Ww2f5HsqHH36o9evX+38/+eST+sMf/qCf/OQnuummmzRt2jRt3LhRL774ohKJhF934YUX+v/OZDJ69NFH20z74Ycf9v8diURaPb9bCgZ3HJKTyaTfkEg7XgPNH4kKhUJOywMAnxsPAL5gc+bM8ST5/91///2t7r/rrrta3f/II4+0+7hjjz221d+nnnqq53me19DQ4B1//PGt7vv0f5MnT/ZyuVyr+Z5zzjnt1p500kkdjvf+++9vdV9LlZWV3mGHHdbhGIqLiz3P87wpU6bsdKySvJUrV3qe53mjR49utQwtPf744148Hu9wGrFYzHv00UdbPWby5Mn+/f379/e6devW7mPvvfde/zGNjY3eoEGDdjrea665xum5UFVV5Q0ePNhpvTevA2s97GybdOTtt99u9Zg777zT6XGe53lDhgzxH3fooYe2ui+ZTHplZWX+/aeffnqr+1sux0EHHeT16dPHk+T16NHDS6fTnud53s9//vNWj+9onQDAF4GPLQHY67z22mut/u7atWu7dfPmzdOwYcN0yimnyPM8/13Zq6++Wv/4xz8kSdFoVOecc44GDhyoRYsWadq0afI8Tw8++KBGjhypK664QpL0xBNPaOrUqf6099tvP5199tlat25dq3eOd8X555+vBQsW+H8PGTJE48ePVywW08KFC/XGG29IksaOHauCggL98Y9/9L8kfuihh7b6iFJZWdlO57Vs2TKdf/75SiaTknZ87Gby5MkKBAJ68MEHtXXrViWTSU2ePFkjR47UwIED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3DjNmjWrTc1+++2n5cuXS5L+8Y9/6D/+4z926xg2b96spUuXSpJ69+6t3r17f+ZprV+/XuPHj9d7773XYc25556rhx56SOFwWNKOZmPBggWSpOLiYh1wwAGfef4AgK+mL9vxtKamRmeddZaef/75DmvuuOMOXXXVVZ95Hv+Ojz/+WJs2bZIkHXDAASouLt4j48AXg+ZhL1JZWamuXbsqk8koHA5r06ZNKisr8+9/5513NHLkSElSRUWF1q1bp1Ao1OH06uvrlZ+f/7mPuz25XE5HH3203njjDUk7dpw///nP1bdvX73++uu6+eab1djYKEn6r//6L/3617/eI+MEAHz5fJmOp5J08skna+bMmZKkeDyuK664wm923n77bd1777264oor9ljzgK8YD3uVE0880ZPkSfL+/Oc/t7rv2muv9e+74oor/NtHjx7t3/722297F154oVdeXu41b95FixZ53/rWt7whQ4Z4paWlXjgc9jp37uyNHz/ee/nll1vN4/777/enNWXKFP/2OXPm+LdPnjzZXI4nn3zSrw+Hw97HH3/c6v4HHnjAvz8ajXrr16/3PM/zVq5c6d8+evRov76hocH70Y9+5O23335eNBr18vLyvL59+3qnn366N3369FbT3rx5s3f11Vf7tSUlJd748eO91157rVVdXV2dd+mll3ojR470unTp4kUiEa+oqMg78sgjvXvuuadV7c6Wv/n2Pn36mOsFAPDF+LIcT2fPnu3XS/KeffbZNjXJZNJbtmyZ//cPf/hD76ijjvK6du3qRaNRLz8/3zv44IO9W2+91Uun060e2/IY9t5773nHHnusl0gkvEGDBnnTpk3zPM/zpk2b5g0dOtSLRqPegQce6L344outpjF58mR/OnPmzGl134svvuiNHz/eKy8v9yKRiNezZ09v8uTJ3tKlS1vVTZkyxZ/Gfffd591xxx3egAEDOpwn9hyah73MQw895L94TjjhhFb3DRgwwL9v/vz5/u0td3b9+/dvtZPxPM979NFHW93W8r9gMOi99NJL/rR2187u29/+tl9/5plntrk/nU57Xbt29Wvuvfdez/M6bh4uuuiiDpfhvPPO8+tWr17t9ezZs926SCTizZgxw6/dsGFDh9OU5N1www1Oy0/zAAB7ny/L8bTl8W/MmDFOyx6LxToc54UXXtiqtvn2kpISv1Fq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dNVVV/3b85GkH//4x7rtttta3ZZKpbRgwQItWLBAd999t5577jkNHTrUv/93v/udtm/fLknq1q3bbhkHAGDfxLFUuuuuu3T11Vcrk8m0uv2DDz7QBx98oFdffVXvvvvubpnXrjr44IP9dVFRUbFHxoDdg+ZhDznxxBP105/+VE1NTXrsscd0zz33SJLuv/9+XXbZZTr00EM7fGwul1MqlWqzI5o4ceJuHeMxxxyzW6fXkd///vd+4xAKhXTVVVfpG9/4hjZv3qxbbrlFixYt0po1a3TKKafovffeU0FBgSTpgAMO+ELGBwDYO3Es/V9PPPGErrjiCv/vsWPH6qKLLlLnzp21evVqPfHEE1q/fv0XMpb2FBcXf2HrAp8vPra0h3Tp0kXHHHOMjj/+eN19993q16+ff19zZ97y0ux9992nX/ziF+rTp48ikYhef/11SZLnebr//vs1atQoFRUVKR6P68ADD9Qdd9yhbDbbZr533XWXBgwYoEQiocMPP1wvvfRSh2P89Oc0m8fTbPXq1e1+lrNv377+7Zampib/3RdJ+r//9//qtttu0/HHH69vfetbevnll1VaWipJWrFihe6++26/tqPvPDz55JM65phjVFxcrGg0qq5du+qYY47RNddc0+pS9qfXXSKR0IgRI3TnnXcql8u1Gue9996rcePGqXfv3srPz1c8HtfAgQN1xRVXaOvWra1qO1r+jj4PCwD4bDiW7pDJZPSf//mf/t9nnnmmZs2apYkTJ+q4447ThRdeqJkzZ+qvf/2rX/P0009rwoQJ6tevnwoLCxWNRtWnTx9deOGFbb5H2PL49dxzz+kHP/iBysvLVVZWpu9///tKJpNas2aNJkyYoIKCAnXt2lXXXXddq2Ppzr7zsHHjRv3gBz/QgAED/I8tjxkzRtOmTWtVt2rVKn8aY8aM0YIFC/T1r39deXl57c4TnxMPX5gpU6Z4kjxJ3uTJk1vdN2LECP++W265pU19//79/X9L8ubMmeN5nudNmjSp1e0t/zvzzDO9XC7nz+PWW29tUxOJRLwhQ4a0ma7nef5tffr0aTOeT//XXON5ntenTx//dss//vEPvzYajXqVlZVtav7zP//Trzn22GP920ePHu3fvnLlSs/zPG/u3LleMBjscJzpdNp//M7W3cSJE1uNYdy4cR3WDhkyxGtsbDSXf/Lkye2uZwCAO46lbb3yyit+bTAY9FasWGE+5rvf/W6H46ioqPA2bdrk17Y8fg0YMKBN/fnnn+/169evze1//vOf/WnMmTOn3e22YsUKr2vXrh2O5ZprrvFrV65c6d/erVs3L5FI7HSe+Hxw5WEPSyaTevjhh/X+++/7t7X3cZwVK1bovPPO08yZM/XQQw+pR48eeuKJJ/TQQw9J2vEOxV/+8hfNmDFDI0aMkLTjEuajjz4qSaqqqtLPf/5zf3pXXHGFZs6cqYkTJ2rx4sVOY73ooov8d3IkqWvXrpo3b57mzZunJ554YtcXXtK//vUv/9+9e/dWWVlZm5qDDjqo3fr2PPvss/67Dr/85S/14osvaurUqbruuus0dOhQ/x2clutu0KBBevTRR/Xss8/6v6rx2GOP6bHHHvOnO3HiRN13332aOXOm5s6dq5kzZ2rSpEmSdny5bvr06Z9h6QEAu8NX/Vj63nvv+f/u0aNHqyswHRk7dqz+9Kc/6dlnn9XcuXM1a9Ys/+rFpk2b/I+AfdrGjRt1991365577lEwuOM08uGHH1ZjY6OmTp2q66+/3q/905/+ZI7jsssu08aNGyXt+ETBM888o//5n//xP07261//Wm+88Uabx23YsEGHHHKIZsyYoR/84Ae7NE/8m/Z09/JVsrN3G5r/O/TQQ71MJtOmftSoUW2md+qpp/r3z5492799/vz5/u3jx4/3PM/zHnvsMf+2ww47zK/NZDJe7969nd4tsW7/LH7xi1/40zvqqKParZk1a5ZfEw6H/dvbu/Jw7bXX+rdNmzbN27p1a7vTbLnufvvb33rz5s3z5s2b5/35z3/2bz/55JP9+jVr1niXXHKJ169fPy8Wi7XZbldffbVfy5UHAPj8cCxtq+Wx9IgjjnB6TGVlpffDH/7QGzRoULvv4J9++ul+bcvj109/+lP/9mHDhvm333vvvZ7neV4ul/MKCws9SV5JSYlf296Vh8rKSi8QCHiSvFgs1uqY3fJTB1deeaXnea2vPESjUW/jxo2e53leNpv18vLy2swTnw++ML2XiEajOvvss/Wb3/xGoVCozf0nn3xym9uWLl3q/3vcuHHtTnfJkiWSdrzb0uywww7z/x0KhTRy5EitWbPmM4/931FUVOT/e8uWLe3WtLy9uLh4p9M777zzdMcddyiZTOqss86StOMzsaNGjdJll12m448/XlLrddfyHYuWmt9Fqq2t1dFHH621a9d2ON/q6uqdjgsA8Pn7qh5LWx4bXb4Unc1mdfzxx2vhwoUd1nR0XDv88MP9f7f8tEDzl9MDgYDKyspUW1trHhs//vhj/7uIAwYMUHl5ebvzabmNmg0ePNj/1aZgMKjS0lI1NDRwPP4C0DzsIc2/EBEIBFRYWKiBAwcqkUh0WG/9rFlRUVG7X6oKh+1N7PJlrM9Ly59eXbNmjaqqqvwvSDdreTm2ZX17hg8frrffflt333233njjDS1ZskSbN2/WU089pRkzZmjevHk6+uijncZWX18vSXrqqaf8xmHw4MG64YYb1L17d7311lu6+uqrJanVF7Rars9sNusfwD79xWoAwL+HY+kOzR+xkqR169Zp1apVOw2lmz9/vt84dOvWTbfccov69eundevW6dxzz5WkDr943LJRaf7YktT6zcDdwVqfnz5XcNlG2D34zsMe0vwLEaNGjdKBBx64052d1P6LaP/99/f/PX36dFVXV7f5rzkQpn///n7tW2+95f87m822+ttF81h2xy8ajBo1Sp06dZK0I9fhrrvuanV/dXV1q9/qPu2003Y6Pc/zNGzYMN155516/fXXVV1d7X+GNJfL6emnn5bUet3NmTNHnue1+W/58uWSduyIm11++eU6++yzdcwxx6ipqandMbTcsTZ/jrO2tlbz58/f6dgBALuGY+kORx11lHr37u1P79prr223rvmKesvj2re+9S1NmjRJxx577L89jl213377+eth+fLlqqys9O9r+T2HltsIex5t2j7svPPO04wZMyRJkyZN0s9+9jMNGjRIW7Zs0dKlS/XMM8/o5JNP1vXXX68TTjhB8XhcTU1NevPNN3XVVVdp3Lhxmjp16i5fZi0tLdW2bdu0fv16PfLII+rTp48qKio0cOBASTu+cLZ69WpJMlM+4/G4fv7zn/sfHbrhhhtUW1ursWPHasuWLbrlllu0bds2f7rf+c53djq9//7v/9bcuXN10kkn+T+rOnv2bP/+ZDLZZt2df/75+tnPfqaBAwdqy5Yt+vjjjzVz5kydeOKJmjJlivr06eM//r777lP//v21bNky/eIXv2h3DPvtt59/tWTSpEk644wz9PDDD3MpFQD2Ql+GY2k4HNZtt92ms88+W9KOH/3Yvn27LrzwwlY5D+vWrdPChQtb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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Image 1 → Vrai: Oiseau, Prédit: Oiseau\n",
            "Image 2 → Vrai: Camion, Prédit: Camion\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# ÉVALUATION AVEC MATRICE DE CONFUSION"
      ],
      "metadata": {
        "id": "p4TG2t3VdCkN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Prédire sur tout le test set\n",
        "y_pred = svm.predict(X_test)\n",
        "\n",
        "# Matrice de confusion\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "TN, FP, FN, TP = cm.ravel()\n",
        "\n",
        "print(\"\\nMATRICE DE CONFUSION:\")\n",
        "print(f\"                  Prédit Oiseau    Prédit Camion\")\n",
        "print(f\"Vrai Oiseau            {TN:4d}            {FP:4d}\")\n",
        "print(f\"Vrai Camion            {FN:4d}            {TP:4d}\")\n",
        "\n",
        "print(f\"\\nDÉTAIL DES VALEURS:\")\n",
        "print(f\"  TN (Vrais Négatifs)  = {TN:4d}  → Oiseaux bien classés\")\n",
        "print(f\"  FP (Faux Positifs)   = {FP:4d}  → Oiseaux classés Camion\")\n",
        "print(f\"  FN (Faux Négatifs)   = {FN:4d}  → Camions classés Oiseau\")\n",
        "print(f\"  TP (Vrais Positifs)  = {TP:4d}  → Camions bien classés\")\n",
        "\n",
        "# Calculer les métriques\n",
        "accuracy = (TP + TN) / (TP + TN + FP + FN)\n",
        "precision = TP / (TP + FP)\n",
        "rappel = TP / (TP + FN)\n",
        "f_mesure = 2 * (precision * rappel) / (precision + rappel)\n",
        "\n",
        "print(f\"\\nMÉTRIQUES DE PERFORMANCE:\")\n",
        "print(f\"  Accuracy   = {accuracy:.4f}  ({accuracy*100:.2f}%)\")\n",
        "print(f\"  Précision  = {precision:.4f}  ({precision*100:.2f}%)\")\n",
        "print(f\"  Rappel     = {rappel:.4f}  ({rappel*100:.2f}%)\")\n",
        "print(f\"  F-mesure   = {f_mesure:.4f}  ({f_mesure*100:.2f}%)\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-Vh5PhszdIKy",
        "outputId": "87eba679-f2f8-4dae-990b-5bf7dfd75dba"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "MATRICE DE CONFUSION:\n",
            "                  Prédit Oiseau    Prédit Camion\n",
            "Vrai Oiseau             833             167\n",
            "Vrai Camion             109             891\n",
            "\n",
            "DÉTAIL DES VALEURS:\n",
            "  TN (Vrais Négatifs)  =  833  → Oiseaux bien classés\n",
            "  FP (Faux Positifs)   =  167  → Oiseaux classés Camion\n",
            "  FN (Faux Négatifs)   =  109  → Camions classés Oiseau\n",
            "  TP (Vrais Positifs)  =  891  → Camions bien classés\n",
            "\n",
            "MÉTRIQUES DE PERFORMANCE:\n",
            "  Accuracy   = 0.8620  (86.20%)\n",
            "  Précision  = 0.8422  (84.22%)\n",
            "  Rappel     = 0.8910  (89.10%)\n",
            "  F-mesure   = 0.8659  (86.59%)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#  ÉTUDE COMPARATIVE: SVM, KNN, DECISION TREE"
      ],
      "metadata": {
        "id": "ynEjm7MHdYX8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Définir les modèles à comparer\n",
        "modeles = {\n",
        "    # SVM avec différents paramètres\n",
        "    'SVM (C=0.1, RBF)': SVC(kernel='rbf', C=0.1, random_state=42),\n",
        "    'SVM (C=1, RBF)': SVC(kernel='rbf', C=1.0, random_state=42),\n",
        "    'SVM (C=10, RBF)': SVC(kernel='rbf', C=10, random_state=42),\n",
        "    'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=42),\n",
        "\n",
        "    # KNN avec différents k\n",
        "    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),\n",
        "    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),\n",
        "    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),\n",
        "    'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),\n",
        "\n",
        "    # Decision Tree avec différentes profondeurs\n",
        "    'DT (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),\n",
        "    'DT (depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42),\n",
        "    'DT (depth=20)': DecisionTreeClassifier(max_depth=20, random_state=42),\n",
        "    'DT (no limit)': DecisionTreeClassifier(random_state=42),\n",
        "}\n",
        "\n",
        "# Entraîner et évaluer chaque modèle\n",
        "resultats = []\n",
        "\n",
        "print(\"\\nEntraînement des modèles...\")\n",
        "for nom, modele in modeles.items():\n",
        "    # Entraîner\n",
        "    modele.fit(X_train, y_train)\n",
        "\n",
        "    # Prédire\n",
        "    y_pred = modele.predict(X_test)\n",
        "\n",
        "    # Calculer les métriques\n",
        "    acc = accuracy_score(y_test, y_pred)\n",
        "    prec = precision_score(y_test, y_pred)\n",
        "    rec = recall_score(y_test, y_pred)\n",
        "    f1 = f1_score(y_test, y_pred)\n",
        "\n",
        "    resultats.append({\n",
        "        'Modèle': nom,\n",
        "        'Accuracy': f\"{acc:.4f}\",\n",
        "        'Précision': f\"{prec:.4f}\",\n",
        "        'Rappel': f\"{rec:.4f}\",\n",
        "        'F-mesure': f\"{f1:.4f}\"\n",
        "    })\n",
        "\n",
        "    print(f\"  ✓ {nom}\")\n",
        "\n",
        "# Créer le tableau des résultats\n",
        "df = pd.DataFrame(resultats)\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"TABLEAU COMPARATIF DES PERFORMANCES\")\n",
        "print(\"=\"*80)\n",
        "print(df.to_string(index=False))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AmQHeW56ddSX",
        "outputId": "aadbaf93-af5b-4fda-8f96-fe1313f82365"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Entraînement des modèles...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ SVM (C=0.1, RBF)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ SVM (C=1, RBF)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ SVM (C=10, RBF)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ SVM (Linear)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/neighbors/_classification.py:239: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
            "  return self._fit(X, y)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ KNN (k=3)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/neighbors/_classification.py:239: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
            "  return self._fit(X, y)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ KNN (k=5)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/neighbors/_classification.py:239: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
            "  return self._fit(X, y)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ KNN (k=7)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/neighbors/_classification.py:239: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
            "  return self._fit(X, y)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  ✓ KNN (k=10)\n",
            "  ✓ DT (depth=5)\n",
            "  ✓ DT (depth=10)\n",
            "  ✓ DT (depth=20)\n",
            "  ✓ DT (no limit)\n",
            "\n",
            "================================================================================\n",
            "TABLEAU COMPARATIF DES PERFORMANCES\n",
            "================================================================================\n",
            "          Modèle Accuracy Précision Rappel F-mesure\n",
            "SVM (C=0.1, RBF)   0.8355    0.8138 0.8700   0.8410\n",
            "  SVM (C=1, RBF)   0.8620    0.8422 0.8910   0.8659\n",
            " SVM (C=10, RBF)   0.8745    0.8563 0.9000   0.8776\n",
            "    SVM (Linear)   0.8300    0.8061 0.8690   0.8364\n",
            "       KNN (k=3)   0.8275    0.8093 0.8570   0.8324\n",
            "       KNN (k=5)   0.8345    0.8083 0.8770   0.8412\n",
            "       KNN (k=7)   0.8345    0.8044 0.8840   0.8423\n",
            "      KNN (k=10)   0.8320    0.8126 0.8630   0.8371\n",
            "    DT (depth=5)   0.8150    0.8137 0.8170   0.8154\n",
            "   DT (depth=10)   0.8060    0.7898 0.8340   0.8113\n",
            "   DT (depth=20)   0.7885    0.7929 0.7810   0.7869\n",
            "   DT (no limit)   0.7820    0.7944 0.7610   0.7773\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Convertir les colonnes de métriques en numériques pour le tri\n",
        "df['Accuracy'] = pd.to_numeric(df['Accuracy'])\n",
        "df['Précision'] = pd.to_numeric(df['Précision'])\n",
        "df['Rappel'] = pd.to_numeric(df['Rappel'])\n",
        "df['F-mesure'] = pd.to_numeric(df['F-mesure'])\n",
        "\n",
        "\n",
        "# Meilleur modèle par métrique\n",
        "best_accuracy = df.loc[df['Accuracy'].idxmax()]\n",
        "best_precision = df.loc[df['Précision'].idxmax()]\n",
        "best_recall = df.loc[df['Rappel'].idxmax()]\n",
        "best_f1 = df.loc[df['F-mesure'].idxmax()]\n",
        "\n",
        "print(\"\\n1. MEILLEURS MODÈLES PAR MÉTRIQUE:\")\n",
        "print(f\"   • Accuracy  : {best_accuracy['Modèle']} ({best_accuracy['Accuracy']:.4f})\")\n",
        "print(f\"   • Précision : {best_precision['Modèle']} ({best_precision['Précision']:.4f})\")\n",
        "print(f\"   • Rappel    : {best_recall['Modèle']} ({best_recall['Rappel']:.4f})\")\n",
        "print(f\"   • F-mesure  : {best_f1['Modèle']} ({best_f1['F-mesure']:.4f})\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QHv8sI8od6S8",
        "outputId": "9f27447c-66ba-4083-8bcf-c3a8d7d8ab12"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "1. MEILLEURS MODÈLES PAR MÉTRIQUE:\n",
            "   • Accuracy  : SVM (C=10, RBF) (0.8745)\n",
            "   • Précision : SVM (C=10, RBF) (0.8563)\n",
            "   • Rappel    : SVM (C=10, RBF) (0.9000)\n",
            "   • F-mesure  : SVM (C=10, RBF) (0.8776)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "Rj5_SBIo2DRb"
      }
    }
  ]
}