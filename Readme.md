AGENTE VIRTUAL

Almabot es una solución digital para luchar contra el bullying, que ofrece recursos y apoyo para la salud mental y el bienestar, incluyendo información sobre ciberacoso y cómo lidiar con situaciones difíciles.

Este ejemplo muestra como utilizar telegram para crear un bot, se utilizará para generar el entorno de pruebas en línea del agente virtual que se esta implementando para la aplicación ALmaBot.

<p>
<img src="https://i.ibb.co/LXbyyxkD/Alma-Chat-Bot.png" alt="Alma-Chat-Bot" />
</p>

> **_LINKS IMPORTANTES:_**<br>
>[Documentación de la API de Telegram Bot](https://core.telegram.org/bots)<br>
>[Documentación telebot](https://github.com/eternnoir/pyTelegramBotAPI)<br>
>[Despliegue de app Almachatbot](https://almachatbot.vercel.app/)

**_Diagrama UML App web:_**<br>

<p>
<img src="https://i.ibb.co/4n4RhsQ5/Diagrama-sin-t-tulo-drawio2-drawio.png" alt="Diagrama UML Alma-Chat-Bot" />
</p>

### Requisitos

Para ejecutar el bot localmente, necesitas lo siguiente:

- Python instalado en tu sistema
- Un bot token de Telegram obtenido del BotFather en Telegram
- La librería `telebot` instalada (`pip install pyTelegramBotAPI`)

> [!NOTE]
> 

🚀 Paso a Paso para Descargar y Ejecutar Almabot

 📂 Resumen de los Pasos
    1. Crear una carpeta donde quieres almacenar el proyecto.

    2. Clonar el repositorio desde GitHub.

    3. Instalar Python y la librería pyTelegramBotAPI.

    4. Crear el archivo .env y poner el token de Telegram.

    5. Ejecutar el bot con el comando python bot.py.

Paso 1:  Crear una Carpeta para el Proyecto

Antes de hacer cualquier cosa, necesitamos crear una carpeta donde vamos a almacenar el código de Almabot.
        1. Abre la terminal o el símbolo del sistema en tu computadora:
          -Si usas Windows, busca CMD o Símbolo del sistema en el menú de inicio (tecla windows+x, seleccionar terminal(administrador)).
          -Si usas Mac, abre Terminal desde las aplicaciones.
          -Si usas Linux, abre Terminal.
        2. Crea una nueva carpeta:

        En la terminal, escribe el siguiente comando para crear una nueva carpeta donde quieras guardar tu proyecto (en este ejemplo vamos a crearla en C:/Proyectos/):


        mkdir C:/Proyectos/Almabot

        Luego, navega dentro de esa carpeta con el comando:

        cd C:/Proyectos/Almabot

        3. Clona el proyecto desde GitHub:

        -Ahora, en la terminal que abriste, escribe este comando (pegando la URL que copiaste) y presiona Enter: 

                https://github.com/LIZGRICAS/Almabot.git

        Esto descargará el proyecto en tu computadora.

Paso 2: Instalar Python y las Librerías Necesarias

        1.Verificar si tienes Python:

        -En la terminal, escribe el siguiente comando y presiona Enter:

        python --version

        Si ves algo como Python 3.x.x, eso significa que ya tienes Python instalado. ¡Genial! Si no, tienes que instalarlo.

        -Si no tienes Python:

        Ve a la página de descarga de Python, descarga la versión más reciente y sigue las instrucciones para instalarlo. Asegúrate de marcar la casilla "Add Python to PATH" durante la instalación.

        2. Instalar las librerías necesarias:

        Una vez que tengas Python instalado, necesitas instalar una librería llamada pyTelegramBotAPI. Esto es como una herramienta para que tu bot funcione.

        En la terminal, escribe el siguiente comando y presiona Enter:

        pip install pyTelegramBotAPI

        Esto descargará la librería necesaria para que el bot se conecte con Telegram.

Paso 3: Crear el Archivo .env y Guardar el Token de Telegram
    1. Abre una ventana nueva de tu visual studio code y con clic sostenido, desplaza la carpeta de tu proyecto a este espacio (tambien la puedes abrir por el menu del visual). o directamente en la carpeta si seguiste el paso 1, la carpeta debería llamarse Almabot, clic derecho y abrir con Vs.

        -Dentro de esa carpeta, crea un archivo nuevo llamado .env. Si estás en Windows, debes crear un archivo que se llame .env sin ningún nombre antes del punto.

    2. Escribir el Token en el archivo .env:

        -Abre el archivo .env con cualquier editor de texto (por ejemplo, Notepad en Windows o TextEdit en Mac).

        -En ese archivo, escribe la siguiente línea, sustituye "TELEGRAM_TOKEN" en el código del bot por tu token bot real.

        TELEGRAM_TOKEN="7968540611:AAGaKKVQcs7Cz9794lIz2AIFwFDtxRq4IFo"
    
    3. Guardar el archivo .env:

    Guarda el archivo y asegúrate de que se guarde con la extensión .env y no como .txt.

Paso 4: Ejecutar el Bot

    1. Ahora que todo está configurado, puedes ejecutar el bot.

    -Abre la terminal (si no la tienes abierta ya).

    -Ve a la carpeta donde está tu proyecto:

    -Si no estás en la carpeta del proyecto, usa el comando cd (cambiar directorio) para llegar allí. Por ejemplo, si el proyecto está en C:/Proyectos/Almabot, escribe:


    cd C:/Proyectos/Almabot

    2. Ejecutar el bot:

    -Ahora, para iniciar el bot, escribe este comando en la terminal:

    python main.py

    ¡Listo! Si todo está bien, tu bot comenzará a funcionar y podrás hablar con él en Telegram.
