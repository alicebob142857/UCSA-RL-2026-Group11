/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  */
/* USER CODE END Header */

#include "main.h"
#include "adc.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"
#include "stdbool.h"

/* USER CODE BEGIN Includes */
#include "sys.h"
/* USER CODE END Includes */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
u8 Pick_up_stop=0;
int Middle_angle=0;
u8 Way_Angle=1;
u16 Flag_front,Flag_back,Flag_Left,Flag_Right,Flag_velocity=2,Target_Velocity=30;
float RC_Velocity,RC_Turn_Velocity;
u8 Flag_Stop=1,Flag_Show=0;
u8 PS2_ON_Flag = 0;
u8 Mode = 0;
float Move_X,Move_Z;
u16 determine;
int Encoder_Left,Encoder_Right;
int Motor_Left,Motor_Right;
int Temperature;
int Voltage;
float Angle_Balance,Gyro_Balance,Gyro_Turn;
u32 Distance;
u8 delay_50,PID_Send;
volatile u8 delay_flag;
u8 Flag_follow=0,Flag_avoid=0,Flag_straight=0;
u8 Lidar_Detect = Lidar_Detect_ON;
float Acceleration_Z;
u8 CCD_Zhongzhi,CCD_Yuzhi;
float Balance_Kp=27000,Balance_Kd=110,Velocity_Kp=400,Velocity_Ki=2,Turn_Kp=4200,Turn_Kd=100;
u16 Angle_ADC = 0;
int Avoid_Flag=0;
int Lidar_Online_Flag=0,Lidar_Online_Cnt=0;
volatile int init_cnt=0,APP_ON=0;
volatile int Steady_Flag,time_cnt=0;

volatile bool answer_flag = true;
volatile bool real_data_stream_enabled = true;
/* USER CODE END PV */

void SystemClock_Config(void);

static void RealData_Telemetry_Task(void)
{
#if Control_mode == 0
  static uint32_t telemetry_tick = 0;
  uint32_t now = HAL_GetTick();
  if ((uint32_t)(now - telemetry_tick) >= REAL_DATA_TELEMETRY_PERIOD_MS) {
    telemetry_tick = now;
    APP_Show();
  }
#endif
}

static void BoardUi_Task(void)
{
#if Control_mode == 0
  static uint32_t oled_tick = 0;
  static uint32_t ps2_tick = 0;
  uint32_t now = HAL_GetTick();

  if ((uint32_t)(now - ps2_tick) >= 20U) {
    ps2_tick = now;
    PS2_Read();
  }
  if ((uint32_t)(now - oled_tick) >= 100U) {
    oled_tick = now;
    oled_show();
  }
#endif
}

int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_TIM3_Init();
  MX_USART1_UART_Init();
  MX_USART3_UART_Init();
  MX_TIM8_Init();
  MX_TIM4_Init();
  MX_ADC2_Init();
  MX_DMA_Init();
  MX_UART4_Init();
  MX_UART5_Init();

  /* USER CODE BEGIN 2 */
  JTAG_Set(JTAG_SWD_DISABLE);
  JTAG_Set(SWD_ENABLE);
  delay_init();
  BEEP_GPIO_Config();
  OLED_Init();
  MPU6050_initialize();
  DMP_Init();
  /* USER CODE END 2 */

  while (1)
  {
#if Control_mode == 0
    Flag_Show = 0;
#endif
    if(Flag_Show==0)
    {
#if Control_mode == 0
      RealData_Telemetry_Task();
      BoardUi_Task();
#else
      if (answer_flag) {
        APP_Show();
        answer_flag = false;
      }
      oled_show();
      PS2_Read();
#endif
    }
    else
    {
      DataScope();
    }

    if(Lidar_Deal_Flag){
      Lidar_data_Deal();
      Lidar_Deal_Flag=0;
    }
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }

  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV6;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif
