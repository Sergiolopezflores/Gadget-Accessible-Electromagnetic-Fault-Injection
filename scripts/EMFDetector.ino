// Required libraries
#include <Wire.h>
#include <LiquidCrystal.h>
#include <SPI.h>
#include <SD.h>

// Sensor parameters
int senseLimit = 4096;
int probePin = 0;
int val = 0;

// Time between readings: 1ms
int updateTime = 1;

// Create LCD instance
LiquidCrystal lcd(7,6,5,8,3,2);

// Create SD instance
File datafile;

void setup()
{
  // Initialize serial connection
  Serial.begin(115200); 
  // Initialize LCD
  lcd.begin(12,2);
  lcd.setCursor(0,0);

  // Display system startup message
  lcd.print("EMF Detector ON");
  delay(1000); 
  // Clear screen
  lcd.clear(); 
  delay(1000);

  // Initialize SD card reader
  SD.begin(10);
  datafile = SD.open("SD.txt", FILE_WRITE);
  if (datafile){
       // If the text file is accessible, display it on the LCD
       lcd.print("SD Card ON");
       datafile.println("Starting data collection....");
    } else {
      // If there was an error accessing the SD card, display it on the LCD
      lcd.print("SD Card ERROR");
      delay(5000);
    }
  datafile.close();
  delay(2000);
}

void loop()
{
  // Read probe data
  val = analogRead(probePin);

  // If we are receiving data
  if(val >= 1){
    // Map obtained data  
    val = map(val, 1, senseLimit, 1, 1023);
    // Display obtained data on LCD
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("EMF level: ");
    lcd.setCursor(1,1);
    lcd.print(val);
    // Create a text file with the malware name
    datafile = SD.open("bashlite.txt", FILE_WRITE);
    if (datafile) {
      // If there was no problem accessing the text file
      datafile.println(val);
      // Close file
      datafile.close();
    } else {
      // If there are problems accessing the text file
      Serial.println("error opening bashlite.txt");      
      }
      // Wait until the next data reading
      delay(updateTime);
  }
}
