// Librerias necesarias
#include <Wire.h>
#include <LiquidCrystal.h>
#include <SPI.h>
#include <SD.h>

// Parámetros del sensor
int senseLimit = 4096;
int probePin = 0;
int val = 0;

//Tiempo entre lecturas 1ms
int updateTime = 1;

//Creamos instancia LCD
LiquidCrystal lcd(7,6,5,8,3,2);

//Creamos instancia SD
File datafile;

void setup()
{
  // Inicializamos conexion serie
  Serial.begin(115200); 
  // Initializamos LCD
  lcd.begin(12,2);
  lcd.setCursor(0,0);

  // Mostramos por pantalla inicio de sistema
  lcd.print("EMF Detector ON");
  delay(1000); 
  // Limpiamos pantalla
  lcd.clear(); 
  delay(1000);

  // Inicializamos lector de tarjeta SD
  SD.begin(10);
  datafile = SD.open("SD.txt", FILE_WRITE);
  if (datafile){
       //Si el fichero de texto es accesible lo mostramos por LCD 
       lcd.print("Tarjeta SD ON");
       datafile.println("Comenzamos toma de datos....");
    } else {
      //Si se ha producido error en el acceso a la tarjeta los mostramos por LCD
      lcd.print("Tarjeta SD KO");
      delay(5000);
    }
  datafile.close();
  delay(2000);
}


void loop()
{
  //lectura de la sonda
  val = analogRead(probePin);

  // Si estamos obteniendo datos
  if(val >= 1){
    //mapeamos datos obtenidos   
    val = map(val, 1, senseLimit, 1, 1023);
    //Mostramos datos obtenidos por LCD
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("EMF level: ");
    lcd.setCursor(1,1);
    lcd.print(val);
    //Creamos fichero txt con nombre de malware
    datafile = SD.open("bashlite.txt", FILE_WRITE);
    if (datafile) {
      //Si no ha habido problema en el acceso al fichero txt
      datafile.println(val);
      //Cerramos fichero
      datafile.close();
    } else {
      //Si hay problemas en el acceso al fichero txt
      Serial.println("error abriendo bashlite.txt");      
      }
      //Esperamos hasta siguiente toma de datos
      delay(updateTime);
  }
}