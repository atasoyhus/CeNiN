using System;

/*
 *--------------------------------------------------------------------------
 * CeNiN > Pool.cs
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
    public class Pool : Layer
    {
        public int[] pool;
        public int[] stride;

        public Pool(int[] inputTensorDims, int[] pad) : base(inputTensorDims, pad)
        {
            type = "Pool";
            pool = new int[2];
            stride = new int[2];
        }

        public new void setOutputDims()
        {
            outputDims = new int[3] {
                (int)Math.Floor((double)InputTensorDims[0] / stride[0]),
                (int)Math.Floor((double)InputTensorDims[1] / stride[1]),
                InputTensorDims[2]
            };
        }

        public override void feedNext()
        {
            outputTensorMemAlloc();

            int inputHeight = InputTensorDims[0];
            int inputWidth = InputTensorDims[1];
            int channelCount = InputTensorDims[2];

            int poolHeight = pool[0];
            int poolWidth = pool[1];

            int[] inputInd = new int[] { 0, 0, 0 };
            int[] outputInd = new int[] { 0, 0, 0 };

            float max;
            for (inputInd[2] = 0; inputInd[2] < channelCount; inputInd[2]++)
            {
                outputInd[2] = inputInd[2];
                for (int i = 0; i <= inputHeight - poolHeight; i += stride[0])
                {
                    outputInd[0] = (int)Math.Floor((double)i / stride[0]);
                    for (int j = 0; j <= inputWidth - poolWidth; j += stride[1])
                    {
                        outputInd[1] = (int)Math.Floor((double)j / stride[1]);

                        max = 0;
                        for (inputInd[0] = i; inputInd[0] < i + poolHeight; inputInd[0]++)
                        {
                            for (inputInd[1] = j; inputInd[1] < j + poolWidth; inputInd[1]++)
                            {
                                float f = inputTensor[inputInd];
                                if (f > max)
                                    max = f;
                            }
                        }
                        writeNextLayerInput(outputInd, max);
                    }
                }
            }

            disposeInputTensor();
        }
    }
}
