#include "fpga_api.h"
#include <stdio.h>
#include <iostream>
#include <cstring>

using namespace std;

#define min(x, y) (((x) < (y)) ? (x) : (y))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;
  data_size_ = (m_size_ + 1) * v_size_; // fpga bram data size

  output_ = new unsigned int[m_size_]; // use output_ as tempolar output
  data_ = new float[data_size_];

  num_block_call_ = 0;
}
FPGA::~FPGA()
{
  delete[] output_;
  delete[] data_;
}

float *FPGA::matrix(void)
{
  return data_ + v_size_;
}

float *FPGA::vector(void)
{
  return data_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

const float *FPGA::blockMV()
{
  num_block_call_ += 1;

  // cpu version
  float *vec = this->vector();
  float *mat = this->matrix();
  float *out = reinterpret_cast<float *>(output_);

  for (int i = 0; i < m_size_; ++i)
  {
    out[i] = 0;
    for (int j = 0; j < v_size_; ++j)
      out[i] += vec[j] * mat[v_size_ * i + j];
  }

  for (int i = 0; i < m_size_; ++i)
    data_[i] = out[i];

  return data_;
}

void FPGA::largeMV(const float *large_mat, const float *input, float *output, int num_input, int num_output)
{
  float *vec = this->vector();
  float *mat = this->matrix();

/*  static int ddd = 0;
  ddd++;
  if(ddd%20 == 0){
  for(int i = 0; i < min(10, num_input); i++)
    printf("%.3f ", input[i]);
  printf("\n%d\n\n", ddd);
  }*/

  // 0) Initialize output vector
  for (int i = 0; i < num_output; ++i)
    output[i] = 0;

  for (int i = 0; i < num_output; i += m_size_)
  {
    for (int j = 0; j < num_input; j += v_size_)
    {
       // 0) Initialize input vector
      int block_row = min(m_size_, num_output - i);
      int block_col = min(v_size_, num_input - j);

      // 1) Assign a vector
      // IMPLEMENT THIS

      memset(vec, 0, sizeof(float)*(block_col+block_row*block_col));

      memcpy(vec, input+j, sizeof(float)*block_col);

      // 2) Assign a matrix
      // IMPLEMENT THIS

      for(int r =0; r < block_row; r++){
        memcpy(mat+r*v_size_, large_mat+(i+r)*num_input+j, sizeof(float)*block_col);
      }

      // 3) Call a function `blockMV() to execute MV multiplication
      const float *ret = this->blockMV();

      // 4) Accumulate intermediate results
      for (int row = 0; row < block_row; ++row)
        output[i + row] += ret[row];
    }
  }
}

void FPGA::convLowering(const std::vector<std::vector<std::vector<std::vector<float>>>> &cnn_weights,
                        std::vector<std::vector<float>> &new_weights,
                        const std::vector<std::vector<std::vector<float>>> &inputs,
                        std::vector<std::vector<float>> &new_inputs)
{
  /*
   * Arguments:
   *
   * conv_weights: [conv_channel, input_channel, conv_height, conv_width]
   * new_weights: [?, ?]
   * inputs: [input_channel, input_height, input_width]
   * new_inputs: [?, ?]
   *
   */
  int conv_channel = cnn_weights.size();
  int input_channel = cnn_weights[0].size();
  int conv_height = cnn_weights[0][0].size();
  int conv_width = cnn_weights[0][0][0].size();
  //int input_channel = inputs.size();
  int input_height = inputs[0].size();
  int input_width = inputs[0][0].size();

  int conv_size = conv_height*conv_width;

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];

  for(int conch = 0; conch < conv_channel; conch++){
    for(int inch = 0; inch < input_channel; inch++){
      for(int i = 0; i < conv_height; i++){
        for(int j = 0; j < conv_width; j++){
          new_weights[conch][inch*conv_size + i*conv_width + j] =
            cnn_weights[conch][inch][i][j];
        }
      }
    }
  }

  int out_height = input_height - conv_height + 1;
  int out_width = input_width - conv_width + 1;

  for(int inch = 0; inch < input_channel; inch++){
    for(int conr = 0; conr < conv_height; conr++){
      for(int conc = 0; conc < conv_width; conc++){
        for(int outr = 0; outr < out_height; outr++){
          for(int outc = 0; outc < out_width; outc++){
            new_inputs[inch*conv_size + conr*conv_width+conc][outr*out_width+outc]
              = inputs[inch][conr+outr][conc+outc];
          }
        }
      }
    }
  }
/*
  static int ccc = 0;
  if(ccc == 100){
    for(int i = 0; i < 6; i++){
      for(int j = 0; j < 8; j++)
        printf("%.3f ", new_weights[i][j]);
      printf("\n");
    }
  }*/
}
