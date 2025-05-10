class UsuarioAnonimo:
    def __init__(self):
        self.id = None
        self.edad = None
        self.barrio = None
        self.colegio = None
        self.sesion_actual = None

class Mensaje:
    def __init__(self, contenido, emisor, timestamp=None):
        self.contenido = contenido
        self.emisor = emisor
        self.timestamp = timestamp or datetime.now()
        self.emocion_detectada = None
        self.bullying_detectado = False

class SesionTerapia:
    def __init__(self, usuario):
        self.usuario = usuario
        self.mensajes = []
        self.fecha_inicio = datetime.now()
        self.fecha_fin = None
        
    def agregar_mensaje(self, mensaje):
        self.mensajes.append(mensaje)
        
    def finalizar(self):
        self.fecha_fin = datetime.now()

class Emocion:
    def __init__(self, nombre, intensidad):
        self.nombre = nombre
        self.intensidad = intensidad 