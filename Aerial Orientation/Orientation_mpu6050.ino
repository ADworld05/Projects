#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// maximum tilt allow before a warning is triggered (in degrees).
const float ROLL_WARNING_THRESHOLD = 30.0; 
const float PITCH_WARNING_THRESHOLD = 30.0; 

void setup() {
  Serial.begin(9600);
  Wire.begin();
  
  Serial.println("Initializing MPU6050...");
  mpu.initialize();
  
  if (mpu.testConnection()) {
    Serial.println("MPU6050 connection successful");
  } else {
    Serial.println("MPU6050 connection failed");
    while (1); 
  }
}

void loop() {
  // This raw accelerometer and gyroscope data.
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  // raw values to pitch, roll, and a basic yaw estimation.
  float pitch = atan2(ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
  float roll = atan2(ay, sqrt(ax * ax + az * az)) * 180.0 / PI;
  float yaw = atan2(az, sqrt(ax * ax + ay * ay)) * 180.0 / PI;
  
    Serial.print("Pitch: ");
  Serial.print(pitch);
  Serial.print(" Roll: ");
  Serial.print(roll);
  Serial.print(" Yaw: ");
  Serial.print(yaw);
  
  // Check for excessive roll and pitch
  if (roll > ROLL_WARNING_THRESHOLD) {
    Serial.print(" WARNING: Tilting too far RIGHT!");
  } else if (roll < -ROLL_WARNING_THRESHOLD) {
    Serial.print(" WARNING: Tilting too far LEFT!");
  }
  
  if (pitch > PITCH_WARNING_THRESHOLD) {
    Serial.print(" WARNING: Nose UP too high!");
  } else if (pitch < -PITCH_WARNING_THRESHOLD) {
    Serial.print(" WARNING: Nose DOWN too low!");
  }
  
  Serial.println(); 
  
  delay(500); 
}