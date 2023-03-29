from pydantic import BaseModel

class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    year: float
    km_driven: float
    owner: object
    seller_type: object
    seats: float
    fuel: object
    transmission: object
    mileage: float
    engine: float
    max_power: float


#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ['year', 'km_driven', 'owner', 'seller_type', 'seats', 'fuel', 'transmission', 'mileage', 'engine', 'max_power']
