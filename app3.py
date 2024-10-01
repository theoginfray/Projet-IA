from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import joblib
import pandas as pd
import logging

logging.basicConfig(filename="server.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Charger le modèle
model = joblib.load("C:\\Users\\theog\\ProjetIA.venv\\.venv\\modele_prediction.pkl")

# Charger les données
data = pd.read_csv('C:\\Users\\theog\\ProjetIA.venv\\.venv\\DATA\\weather_prediction_dataset2.csv', delimiter=';')

# Convertir les données du format large au format long
df_long_corrected = pd.melt(data, 
                            id_vars=['DATE', 'MONTH'], 
                            var_name='Ville', 
                            value_name='Temperature')

df_long_corrected.Ville = df_long_corrected.Ville.str.replace("_temp_mean", "")
df_long_corrected.DATE = pd.to_datetime(df_long_corrected.DATE, format='%Y%m%d')

data = df_long_corrected.dropna(axis=0)
data['Ville_code'] = pd.factorize(data['Ville'])[0] + 1

data['year'] = pd.to_datetime(data['DATE']).dt.year
data['month'] = pd.to_datetime(data['DATE']).dt.month
data['day'] = pd.to_datetime(data['DATE']).dt.day

# Liste des villes et leurs codes
villes_list = data[['Ville', 'Ville_code']].drop_duplicates().set_index('Ville').to_dict()['Ville_code']

class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            logging.info("Requête GET reçue pour l'interface utilisateur")
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open("C:\\Users\\theog\\ProjetIA.venv\\.venv\\index.html", "rb") as file:
                self.wfile.write(file.read())
        elif self.path == "/villes":
            logging.info("Requête GET reçue pour la liste des villes")
            self._set_headers()
            self.wfile.write(json.dumps({"Ville": list(villes_list.keys())}).encode('utf-8'))
        else:
            logging.warning("Requête GET reçue pour une URL non trouvée : %s", self.path)
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            logging.info("Requête POST reçue avec les données : %s", data)

            # Récupérer les valeurs pour l'année, le mois, le jour, et la ville
            year = int(data['year'])
            month = int(data['month'])
            day = int(data['day'])
            ville = data['ville']
            logging.info(f"Année: {year}, Mois: {month}, Jour: {day}, Ville: {ville}")

            # Convertir la ville en code numérique pour le modèle
            if ville in villes_list:
                ville_code = villes_list[ville]
                logging.info(f"Code de la ville pour {ville} est {ville_code}")
            else:
                raise ValueError(f"Ville '{ville}' non trouvée dans la liste")

            # Effectuer la prédiction
            input_data = [[year, month, day, ville_code]]
            logging.info(f"Données d'entrée pour la prédiction: {input_data}")
            
            prediction = model.predict(input_data)
            logging.info(f"Prédiction effectuée : {prediction[0]}")

            # Envoyer la réponse JSON
            self._set_headers()
            self.wfile.write(json.dumps({'predicted_temperature': round(prediction[0], 2)}).encode('utf-8'))

        except Exception as e:
            logging.error("Erreur lors du traitement de la requête POST : %s", str(e))
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info("Serveur démarré sur le port %s", port)
    print(f"Server running on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.critical("Le serveur s'est arrêté en raison d'une erreur : %s", str(e))
