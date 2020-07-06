# ORIprojekat
ORI  
Marija Ćurčić SW24/2017  
Sara Miketek SW62/2017  
Jovana Kostreš SW51/2017  
  
# Pacman projekat
Algoritmi koji su korišćeni:   
Offensive agent - Expectimax  
Defensive agent - Redefinisan refleksni agent    

Fajl za pokretanje - capture.py

# Projekat grupisanja korisnika kreditnih kartica  
Algoritam koji je korišćen za klasterizaciju - klasterizacija metodom K-srednjih vrednosti  

Fajl za pokretanje - main.py

# Klasifikacija snimaka  
Korišćeni paketi tensorflow, keras i matplotlib za izradu algoritma
  
Učitavanje podataka za treniranje - training_data.py  
Učitavanje podataka za testiranja - test_data.py  
Fajl za pokretanje - cnn.py

Isprobane su četiri različite arhitekture mreže, od kojih se najbolje pokazala mreža koja sadrži dva konvoluciona sloja sa aktivacionom funkciom ReLU, dva MaxPooling sloja i Softmax na kraju. Za regularizaciju je korišćen Dropout. Trenirana je u 10 epoha. Validation accuracy za testne podatke iznosi oko 0.8.
