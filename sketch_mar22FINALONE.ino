// PWM引脚定义 PWM-FORWARD DIGITAL-BACKWARD
const int leftFrontPWM = 3;    // PWM控制左前轮速度
const int leftFrontDir = 4;    // 控制左前轮方向
const int rightFrontPWM = 5;   // PWM控制右前轮速度
const int rightFrontDir = 7;   // 控制右前轮方向
const int leftBackPWM = 9;     // PWM控制左后轮速度
const int leftBackDir = 12;     // 控制左后轮方向
const int rightBackPWM = 6;    // PWM控制右后轮速度
const int rightBackDir = 8;   // 控制右后轮方向

const int motorSpeed = 50;     // 电机速度（0-255）

void setup() {
  // 设置所有引脚为输出
  pinMode(leftFrontPWM, OUTPUT);
  pinMode(leftFrontDir, OUTPUT);
  pinMode(rightFrontPWM, OUTPUT);
  pinMode(rightFrontDir, OUTPUT);
  pinMode(leftBackPWM, OUTPUT);
  pinMode(leftBackDir, OUTPUT);
  pinMode(rightBackPWM, OUTPUT);
  pinMode(rightBackDir, OUTPUT);
  
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch(command) {
      case 'F':
        moveForward();
        Serial.println("Moving Forward");
        break;
      case 'L':
        turnLeft();
        Serial.println("Turning Left");
        break;
      case 'R':
        turnRight();
        Serial.println("Turning Right");
        break;
      case 'S':
        stopMotors();
        Serial.println("Stopping");
        break;
      case 'B':
        moveBackward();
        Serial.println("Moving Backward");
        break;
    }
  }
}

void moveForward() {
  // 左前轮前进
  analogWrite(leftFrontPWM, motorSpeed);
  digitalWrite(leftFrontDir, LOW);
  
  // 右前轮前进
  analogWrite(rightFrontPWM, motorSpeed);
  digitalWrite(rightFrontDir, LOW);
  
  // 左后轮前进
  analogWrite(leftBackPWM, motorSpeed);
  digitalWrite(leftBackDir, LOW);
  
  // 右后轮前进
  analogWrite(rightBackPWM, motorSpeed);
  digitalWrite(rightBackDir, LOW);
}

void turnLeft() {
  // 左侧电机后退
  digitalWrite(leftFrontDir, LOW);
  analogWrite(leftFrontPWM, 0);
  digitalWrite(leftBackDir, LOW);
  analogWrite(leftBackPWM, 0);
  
  // 右侧电机前进
  digitalWrite(rightFrontDir, LOW);
  analogWrite(rightFrontPWM, 150);
  digitalWrite(rightBackDir, LOW);
  analogWrite(rightBackPWM, 150);
}

void turnRight() {
  // 左侧电机前进
  digitalWrite(leftFrontDir, LOW);
  analogWrite(leftFrontPWM, 150);
  digitalWrite(leftBackDir, LOW);
  analogWrite(leftBackPWM, 150);
  
  // 右侧电机后退
  digitalWrite(rightFrontDir,LOW);
  analogWrite(rightFrontPWM, 0);
  digitalWrite(rightBackDir, LOW);
  analogWrite(rightBackPWM, 0);
}

void stopMotors() {
  // 停止所有电机
  analogWrite(leftFrontPWM, 0);
  analogWrite(rightFrontPWM, 0);
  analogWrite(leftBackPWM, 0);
  analogWrite(rightBackPWM, 0);
  digitalWrite(leftFrontDir, LOW);
  digitalWrite(rightFrontDir, LOW);
  digitalWrite(leftBackDir, LOW);
  digitalWrite(rightBackDir, LOW);
}
void moveBackward() {
  analogWrite(leftFrontPWM, 0);
  digitalWrite(leftFrontDir, HIGH);
  
  analogWrite(rightFrontPWM, motorSpeed);
  digitalWrite(rightFrontDir, LOW);
  
  analogWrite(leftBackPWM, 0);
  digitalWrite(leftBackDir, HIGH);
  
  analogWrite(rightBackPWM, motorSpeed);
  digitalWrite(rightBackDir, LOW);
}
