import numpy as np
from sklearn import neighbors
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def HarfleriKucult(Yazi):
    return Yazi.replace('İ','i').lower()
def GereksizNoktalama(Yazi):
    Yazi = Yazi.replace('.',' ')
    Yazi = Yazi.replace(',',' ')
    Yazi = Yazi.replace('-',' ')
    Yazi = Yazi.replace(';',' ')
    Yazi = Yazi.replace(':',' ')
    return Yazi
def KelimeleriAyir(Yazi):
    return Yazi.split(' ')

def SorunuCoz(kelimeler):
    sayi = 0
    for t in range(len(kelimeler)):
        if(kelimeler[t]==''):sayi+=1;
    for t in range(sayi):kelimeler.remove('')
    return kelimeler

# Bir metin verince onun 0101101 şekline getirir
def BilgiCikarimiYap(Yazi,kelimeler):
    Yazi = HarfleriKucult(Yazi)
    Yazi = GereksizNoktalama(Yazi)
    YazıdakiKelimeler = KelimeleriAyir(Yazi)
    YazıdakiKelimeler = SorunuCoz(YazıdakiKelimeler)

    Sonuç = np.empty(kelimeler.size)#benim elimdeki sözlüğe bakıp 0-1 vercem o yüzden sözlük boyutu kadar
    for t in range(kelimeler.size):
        if(kelimeler[t] in YazıdakiKelimeler):
            Sonuç[t]= 1
        else:
            Sonuç[t]= 0
    return Sonuç

#elimizde içinde cümleler olan X dosyası var
# bu cümlelerden sözlük yapacaz
def KelimeListesiniOluştur(X):
    Sonuç = []
    for t in range(X.size):
        Yazi = HarfleriKucult(X[t])
        Yazi = GereksizNoktalama(Yazi)
        YazıdakiKelimeler = KelimeleriAyir(Yazi)
        YazıdakiKelimeler = SorunuCoz(YazıdakiKelimeler)

        for t in range(len(YazıdakiKelimeler)):
            if(YazıdakiKelimeler not in Sonuç):
                Sonuç.append(YazıdakiKelimeler[t])

    return np.array(Sonuç)

def VeriyiUygunHaleGetir(X,KelimeListesi):
    Sonuç=[]
    for t in range(X.size):
        Sonuç.append(BilgiCikarimiYap(X[t],KelimeListesi))
    return np.array(Sonuç)



x = np.load('X.npy')
y = np.load('y.npy')

KelimeListesi=KelimeListesiniOluştur(x)
x = VeriyiUygunHaleGetir(x,KelimeListesi)

#tanımlıyoruz
Sınıflandırıcı_kNN = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance')
Sınıflandırıcı_NaivBayes = GaussianNB()
Sınıflandırıcı_DecisionTree = tree.DecisionTreeClassifier()
Sınıflandırıcı_SVM = SVC()

#eğitiyoruz
Sınıflandırıcı_kNN.fit(x,y)
Sınıflandırıcı_NaivBayes.fit(x,y)
Sınıflandırıcı_DecisionTree.fit(x,y)
Sınıflandırıcı_SVM.fit(x,y)

Yazi = input('Yazi Girin : ')
İslenmisYazi = BilgiCikarimiYap(Yazi,KelimeListesi)
#predict: TAHMİN ET -- eğitilmiş bir yapı var sen ona denemek için bir girdi veriyosun elle
print('kNN : '+ str(Sınıflandırıcı_kNN.predict([İslenmisYazi])))
print('NaivBayes : '+ str(Sınıflandırıcı_NaivBayes.predict([İslenmisYazi])))
print('DecisionTree : '+ str(Sınıflandırıcı_DecisionTree.predict([İslenmisYazi])))
print('SVM : '+ str(Sınıflandırıcı_SVM.predict([İslenmisYazi])))




