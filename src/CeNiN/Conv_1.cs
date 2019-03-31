using System;

/*
 *--------------------------------------------------------------------------
 * CeNiN > Conv_1.cs
 *--------------------------------------------------------------------------
 * CeNiN; a convolutional neural network implementation in pure C#
 * Huseyin Atasoy
 * huseyin @atasoyweb.net
 * http://huseyinatasoy.com
 * March 2019
 *--------------------------------------------------------------------------
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *--------------------------------------------------------------------------
 */

namespace CeNiN
{
    public class Conv_1 : Layer
    {
        public int[] stride;

        public Tensor weights;
        public Tensor biases;

        public Conv_1(int[] inputTensorDims, int[] pad) : base(inputTensorDims, pad)
        {
            type = "Convolution";
            stride = new int[2];
        }

        public new void setOutputDims()
        {
            int newHeight = (int)Math.Floor((double)(InputTensorDims[0] - weights.Dimensions[0]) / stride[0]) + 1;
            int newWidth = (int)Math.Floor((double)(InputTensorDims[1] - weights.Dimensions[1]) / stride[1]) + 1;

            outputDims = new int[] { newHeight, newWidth, weights.Dimensions[3] };
        }

        public override void feedNext()
        {
            outputTensorMemAlloc();

            int inputHeight = InputTensorDims[0];
            int inputWidth = InputTensorDims[1];
            int filterHeight = weights.Dimensions[0];
            int filterWidth = weights.Dimensions[1];
            int channelCount = weights.Dimensions[2];
            int filterCount = weights.Dimensions[3];

            int[] weightsInd = new int[] { 0, 0, 0, 0 };
            int[] inputInd = new int[] { 0, 0, 0 };
            int[] outputInd = new int[] { 0, 0, 0 };

            for (int i = 0; i <= inputHeight - filterHeight; i += stride[0])
            {
                outputInd[0] = (int)Math.Floor((double)i / stride[0]);
                for (int j = 0; j <= inputWidth - filterWidth; j += stride[1])
                {
                    outputInd[1] = (int)Math.Floor((double)j / stride[1]);
                    for (weightsInd[3] = 0; weightsInd[3] < filterCount; weightsInd[3]++)
                    {
                        outputInd[2] = weightsInd[3];
                        float sum = 0;
                        for (inputInd[2] = 0; inputInd[2] < channelCount; inputInd[2]++)
                        {
                            weightsInd[2] = inputInd[2];
                            for (inputInd[0] = i; inputInd[0] < i + filterHeight; inputInd[0]++)
                            {
                                weightsInd[0] = inputInd[0] - i;
                                for (inputInd[1] = j; inputInd[1] < j + filterWidth; inputInd[1]++)
                                {
                                    weightsInd[1] = inputInd[1] - j;
                                    sum += inputTensor[inputInd] * weights[weightsInd];
                                }
                            }
                        }
                        sum += biases[new int[] { weightsInd[3] }];
                        writeNextLayerInput(outputInd, sum);
                    }
                }
            }

            disposeInputTensor();
        }
    }
}
