using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

/*
 *--------------------------------------------------------------------------
 * CeNiN > CNN.cs
 *--------------------------------------------------------------------------
 * CeNiN; a convolutional neural network implementation in pure C#
 * Huseyin Atasoy
 * huseyin @atasoyweb.net
 * http://huseyinatasoy.com
 * March 2019
 *--------------------------------------------------------------------------
 * Copyright 2019 Huseyin Atasoy
 *
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
    unsafe public class CNN
    {
        private const string CeNiN_FILE_HEADER = "CeNiN NEURAL NETWORK FILE";

        public int layerCount;
        public int classCount;
        public int totalWeightCount;
        public int totalBiasCount;

        public InputLayer inputLayer;

        private Layer[] layers;

        public CNN(string path)
        {
            FileStream f = null;
            BinaryReader br = null;
            try
            {
                f = new FileStream(path, FileMode.Open);
                br = new BinaryReader(f, Encoding.ASCII, false);
                char[] c = br.ReadChars(25);
                if (!(new string(c)).Equals(CeNiN_FILE_HEADER))
                    throw new Exception("Invalid file header!");

                layerCount = br.ReadInt32();
                int[] inputSize = new int[3];
                for (int i = 0; i < 3; i++)
                    inputSize[i] = br.ReadInt32();

                inputLayer = new InputLayer(inputSize);
                inputLayer.layerCount = layerCount;

                for (int i = 0; i < 3; i++)
                    inputLayer.avgPixel[i] = br.ReadSingle();

                inputLayer.setOutputDims();

                Layer layerChain = inputLayer;
                Layer currentLayer = inputLayer;

                totalWeightCount = 0;
                totalBiasCount = 0;

                List<Layer> layerList = new List<Layer>();
                layerList.Add(currentLayer);

                bool endOfFile = false;
                while (!endOfFile)
                {
                    string layerT = br.ReadString();
                    if (layerT.Equals("conv"))
                    {
                        int[] pad = new int[4];
                        for (int i = 0; i < 4; i++)
                            pad[i] = br.ReadByte();

                        int[] inputTensorDims = currentLayer.outputDims;
                        Conv cLayer = new Conv(inputTensorDims, pad);

                        int[] dims = new int[4];
                        for (int i = 0; i < 4; i++)
                            dims[i] = br.ReadInt32();

                        for (int i = 0; i < 2; i++)
                            cLayer.stride[i] = br.ReadByte();

                        cLayer.weights = new Tensor(dims);
                        for (int i = 0; i < cLayer.weights.TotalLength; i++)
                            cLayer.weights.memPtr[i] = br.ReadSingle();
                        totalWeightCount += cLayer.weights.TotalLength;

                        cLayer.biases = new Tensor(new int[] { dims[3] });
                        for (int i = 0; i < cLayer.biases.TotalLength; i++)
                            cLayer.biases.memPtr[i] = br.ReadSingle();
                        totalBiasCount += cLayer.biases.TotalLength;

                        cLayer.setOutputDims();

                        currentLayer = cLayer;
                    }
                    else if (layerT.Equals("relu"))
                    {
                        ReLU rLayer = new ReLU(currentLayer.outputDims);
                        rLayer.setOutputDims();
                        currentLayer = rLayer;
                    }
                    else if (layerT.Equals("pool"))
                    {
                        int[] pad = new int[4];
                        for (int i = 0; i < 4; i++)
                            pad[i] = br.ReadByte();

                        Pool pLayer = new Pool(currentLayer.outputDims, pad);

                        for (int i = 0; i < 2; i++)
                            pLayer.pool[i] = br.ReadByte();

                        for (int i = 0; i < 2; i++)
                            pLayer.stride[i] = br.ReadByte();

                        pLayer.setOutputDims();
                        currentLayer = pLayer;
                    }
                    else if (layerT.Equals("softmax"))
                    {
                        classCount = br.ReadInt32();
                        string[] classes = new string[classCount];
                        for (int i = 0; i < classCount; i++)
                            classes[i] = br.ReadString();

                        SoftMax smLayer = new SoftMax(currentLayer.outputDims);
                        currentLayer.appendNext(smLayer);
                        Output oLayer = new Output(smLayer.InputTensorDims, classes);
                        smLayer.appendNext(oLayer);
                        layerList.Add(smLayer);
                        layerList.Add(oLayer);
                        continue;
                    }
                    else if (layerT.Equals("EOF"))
                    {
                        endOfFile = true;
                        continue;
                    }
                    else
                        throw new Exception("The following layer is not implemented: " + layerT);

                    layerList.Add(currentLayer);
                    layerChain.appendNext(currentLayer);
                    layerChain = layerChain.nextLayer;
                }
                layers = layerList.ToArray();
            }
            catch (Exception e)
            {
            }
            finally
            {
                if (br != null) br.Close();
                if (f != null) f.Close();
            }
        }
    }
}
