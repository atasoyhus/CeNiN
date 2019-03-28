using System;
using System.Windows.Forms;
using CeNiN;
using System.Drawing;

/*
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

namespace CeNiN_CSharp_Example
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        DateTime dateTime;

        CNN cnn;

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog opf = new OpenFileDialog();
            opf.Filter = "CeNiN file|*.cenin";
            if (opf.ShowDialog() != DialogResult.OK) return;

            textBox1.Clear();
            prependLine("Parsing CeNiN file...");
            Application.DoEvents();
            tic();
            cnn = new CNN(opf.FileName);
            prependLine("Neural network loaded sucesfully in " + toc() + " seconds.");
            prependLine(cnn.layerCount + "+2 layers, "
                    + cnn.totalWeightCount + " weights and"
                    + cnn.totalBiasCount + " biases were loaded in "
                    + toc() + " seconds.");

            button2.Enabled = true;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog opf = new OpenFileDialog();
            opf.Filter = "Image files|*.bmp;*.jpeg;*.jpg;*.png";
            if (opf.ShowDialog() != DialogResult.OK) return;
            Bitmap b = new Bitmap(opf.FileName);
            pictureBox1.Image = b;
            cnn.inputLayer.setInput(b);
            tic();
            Layer currentLayer = cnn.inputLayer;
            int i = 0;
            while (currentLayer.nextLayer != null)
            {
                if (i == 0)
                    prependLine("Loading bitmap data...");
                else
                    prependLine("Layer " + i + " (" + currentLayer.type + ") ...");

                Application.DoEvents();

                currentLayer.feedNext();
                currentLayer = currentLayer.nextLayer;
                i += 1;
            }

            Output outputLayer = (Output)currentLayer;
            prependLine("Finished in " + toc().ToString() + " seconds");

            if (b.Width != b.Height)
                prependLine("WARNING: Since aspect ratio of the network input is 1:1, the image was stretched and this may have affected the result negatively.");

            string decision = outputLayer.getDecision();
            string hLine = new string('-', 100);
            prependLine(hLine, "");
            for (i = 2; i >= 0; i--)
                prependLine(" #" + (i + 1) + "   " + outputLayer.sortedClasses[i] + " (" + Math.Round(outputLayer.probabilities[i], 3) + ")", "");
            prependLine(hLine, "");
            prependLine("THE HIGHEST 3 PROBABILITIES: ", "");
            prependLine(hLine, "");
            prependLine("DECISION: " + decision);
            prependLine(hLine, "");
        }

        private DateTime tic()
        {
            dateTime = DateTime.Now;
            return dateTime;
        }
        private double toc()
        {
            return Math.Round((DateTime.Now - dateTime).TotalSeconds, 3);
        }

        private void prependLine(string text, string prefix = "-->  ")
        {
            textBox1.Text = prefix + text + "\r\n" + textBox1.Text;
        }
    }
}
