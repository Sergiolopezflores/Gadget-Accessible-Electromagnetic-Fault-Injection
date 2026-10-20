#include <Arduino.h>
#include <SPI.h>
#include <SD.h>

// ============================================================
// CONFIGURACIÓN
// ============================================================

// Entrada analógica donde está conectada la sonda
const uint8_t PROBE_PIN = A0;

// Chip Select de la tarjeta SD
const uint8_t SD_CS_PIN = 10;

// Cada cuántas muestras forzamos escritura a SD
const uint16_t FLUSH_EVERY = 100;

// Duración de cada captura: 15 minutos
const uint32_t FILE_DURATION_MS =
    15UL * 60UL * 1000UL;

// Número de capturas por ejecución/run
const uint8_t CAPTURES_PER_RUN = 2;

// Archivo donde guardamos el último run_id
const char RUN_ID_FILE[] = "RUNID.TXT";


// ============================================================
// VARIABLES
// ============================================================

File datafile;

uint32_t sampleCount = 0;
uint32_t previousTimestamp = 0;

uint32_t fileStartTime = 0;

// ID de la ejecución actual
uint16_t runId = 0;

// Captura dentro del run:
// 1, 2
uint8_t captureInRun = 1;


// ============================================================
// OBTENER NUEVO RUN ID
// ============================================================

uint16_t getNextRunId()
{
    uint16_t lastRunId = 0;

    // --------------------------------------------------------
    // Leer último run_id si existe
    // --------------------------------------------------------

    if (SD.exists(RUN_ID_FILE))
    {
        File runFile = SD.open(
            RUN_ID_FILE,
            FILE_READ
        );

        if (runFile)
        {
            lastRunId = runFile.parseInt();
            runFile.close();
        }
    }

    // Nuevo run
    uint16_t newRunId = lastRunId + 1;


    // --------------------------------------------------------
    // Guardar nuevo run_id
    // --------------------------------------------------------

    if (SD.exists(RUN_ID_FILE))
    {
        SD.remove(RUN_ID_FILE);
    }

    File runFile = SD.open(
        RUN_ID_FILE,
        FILE_WRITE
    );

    if (!runFile)
    {
        Serial.println(
            "ERROR: Could not save RUN ID."
        );

        while (1)
        {
        }
    }

    runFile.println(newRunId);
    runFile.flush();
    runFile.close();

    return newRunId;
}


// ============================================================
// CREAR NUEVA CAPTURA
// ============================================================

void openNewFile()
{
    // --------------------------------------------------------
    // Cerrar archivo anterior
    // --------------------------------------------------------

    if (datafile)
    {
        datafile.flush();
        datafile.close();
    }


    // --------------------------------------------------------
    // Nombre:
    //
    // R01C01.TXT
    // R01C02.TXT
    // R02C01.TXT
    // ...
    //
    // Compatible con nombres FAT 8.3
    // --------------------------------------------------------

    char fileName[13];

    snprintf(
        fileName,
        sizeof(fileName),
        "R%02uC%02u.TXT",
        runId,
        captureInRun
    );


    // --------------------------------------------------------
    // Seguridad: NO sobrescribir
    // --------------------------------------------------------

    if (SD.exists(fileName))
    {
        Serial.print(
            "ERROR: File already exists: "
        );

        Serial.println(fileName);

        while (1)
        {
        }
    }


    // --------------------------------------------------------
    // Abrir archivo
    // --------------------------------------------------------

    datafile = SD.open(
        fileName,
        FILE_WRITE
    );

    if (!datafile)
    {
        Serial.print(
            "ERROR: Could not open file: "
        );

        Serial.println(fileName);

        while (1)
        {
        }
    }


    // --------------------------------------------------------
    // Cabecera
    //
    // Incluimos run_id y capture_id también dentro del CSV
    // --------------------------------------------------------

    datafile.println(
        "run_id,capture_id,sample,t_us,dt_us,adc_raw"
    );

    datafile.flush();


    // --------------------------------------------------------
    // Reiniciar variables
    // --------------------------------------------------------

    sampleCount = 0;
    previousTimestamp = 0;

    fileStartTime = millis();


    // --------------------------------------------------------
    // Información Serial
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.print("RUN ID: ");
    Serial.println(runId);

    Serial.print("CAPTURE: ");
    Serial.print(captureInRun);
    Serial.print("/");
    Serial.println(CAPTURES_PER_RUN);

    Serial.print("FILE: ");
    Serial.println(fileName);

    Serial.print("Duration: ");
    Serial.print(FILE_DURATION_MS / 60000UL);
    Serial.println(" minutes");

    Serial.println(
        "=========================================="
    );
    Serial.println();
}


// ============================================================
// FINALIZAR RUN
// ============================================================

void finishRun()
{
    if (datafile)
    {
        datafile.flush();
        datafile.close();
    }

    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.print("RUN ");
    Serial.print(runId);
    Serial.println(" FINISHED");

    Serial.print(CAPTURES_PER_RUN);
    Serial.println(
        " captures completed."
    );

    Serial.println();
    Serial.println(
        "ACQUISITION STOPPED."
    );

    Serial.println(
        "Restart/reset Arduino to begin a new run."
    );

    Serial.println(
        "=========================================="
    );


    // --------------------------------------------------------
    // Detener Arduino aquí
    // --------------------------------------------------------

    while (1)
    {
        delay(1000);
    }
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    delay(2000);

    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "EM-SENSE - ADC + SD ACQUISITION"
    );

    Serial.println(
        "Arduino UNO R4 WiFi"
    );

    Serial.println(
        "=========================================="
    );

    Serial.println();


    // --------------------------------------------------------
    // Inicializar SD
    // --------------------------------------------------------

    Serial.println(
        "Initializing SD card..."
    );

    if (!SD.begin(SD_CS_PIN))
    {
        Serial.println(
            "ERROR: SD initialization failed."
        );

        while (1)
        {
        }
    }

    Serial.println(
        "SD initialized correctly."
    );


    // --------------------------------------------------------
    // Obtener run_id
    // --------------------------------------------------------

    runId = getNextRunId();

    Serial.println();

    Serial.print(
        "New RUN ID: "
    );

    Serial.println(runId);


    // --------------------------------------------------------
    // Información
    // --------------------------------------------------------

    Serial.println();
    Serial.println("Configuration:");

    Serial.println(
        "  ADC input: A0"
    );

    Serial.print(
        "  Capture duration: "
    );

    Serial.print(
        FILE_DURATION_MS / 60000UL
    );

    Serial.println(
        " minutes"
    );

    Serial.print(
        "  Captures per run: "
    );

    Serial.println(
        CAPTURES_PER_RUN
    );

    Serial.print(
        "  Total run duration: "
    );

    Serial.print(
        (
            FILE_DURATION_MS *
            CAPTURES_PER_RUN
        ) / 60000UL
    );

    Serial.println(
        " minutes"
    );

    Serial.print(
        "  Flush every: "
    );

    Serial.print(
        FLUSH_EVERY
    );

    Serial.println(
        " samples"
    );


    Serial.println();
    Serial.println(
        "Starting acquisition in 2 seconds..."
    );

    delay(2000);


    // --------------------------------------------------------
    // Primera captura
    // --------------------------------------------------------

    captureInRun = 1;

    openNewFile();

    Serial.println(
        "ACQUISITION STARTED"
    );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    // --------------------------------------------------------
    // ¿Han terminado los 15 minutos?
    // --------------------------------------------------------

    if (
        (uint32_t)(
            millis() - fileStartTime
        ) >= FILE_DURATION_MS
    )
    {
        // Cerrar captura actual
        datafile.flush();
        datafile.close();


        // ----------------------------------------------------
        // Si ya tenemos las 2 capturas:
        // terminar completamente este run
        // ----------------------------------------------------

        if (
            captureInRun >=
            CAPTURES_PER_RUN
        )
        {
            finishRun();
        }


        // ----------------------------------------------------
        // Si no, empezar segunda captura
        // ----------------------------------------------------

        captureInRun++;

        openNewFile();
    }


    // --------------------------------------------------------
    // Timestamp
    // --------------------------------------------------------

    uint32_t timestamp = micros();


    // --------------------------------------------------------
    // ADC
    // --------------------------------------------------------

    uint16_t adcValue =
        analogRead(PROBE_PIN);


    // --------------------------------------------------------
    // dt
    // --------------------------------------------------------

    uint32_t dt = 0;

    if (previousTimestamp != 0)
    {
        dt =
            timestamp -
            previousTimestamp;
    }

    previousTimestamp = timestamp;


    // --------------------------------------------------------
    // Contador
    // --------------------------------------------------------

    sampleCount++;


    // --------------------------------------------------------
    // Guardar:
    //
    // run_id,capture_id,sample,t_us,dt_us,adc_raw
    // --------------------------------------------------------

    datafile.print(runId);
    datafile.print(',');

    datafile.print(captureInRun);
    datafile.print(',');

    datafile.print(sampleCount);
    datafile.print(',');

    datafile.print(timestamp);
    datafile.print(',');

    datafile.print(dt);
    datafile.print(',');

    datafile.println(adcValue);


    // --------------------------------------------------------
    // Flush
    // --------------------------------------------------------

    if (
        (sampleCount % FLUSH_EVERY)
        == 0
    )
    {
        datafile.flush();

        Serial.print(
            "Run "
        );

        Serial.print(
            runId
        );

        Serial.print(
            " | Capture "
        );

        Serial.print(
            captureInRun
        );

        Serial.print(
            " | Samples: "
        );

        Serial.println(
            sampleCount
        );
    }
}