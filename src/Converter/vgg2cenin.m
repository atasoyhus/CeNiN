
 %--------------------------------------------------------------------------
 % CeNiN; a convolutional neural network implementation in pure C#
 % Huseyin Atasoy
 % huseyin @atasoyweb.net
 % http://huseyinatasoy.com
 % March 2019
 %--------------------------------------------------------------------------
 % Copyright 2019 Huseyin Atasoy
 %
 % Licensed under the Apache License, Version 2.0 (the "License");
 % you may not use this file except in compliance with the License.
 % You may obtain a copy of the License at
 %
 % http://www.apache.org/licenses/LICENSE-2.0
 %
 % Unless required by applicable law or agreed to in writing, software
 % distributed under the License is distributed on an "AS IS" BASIS,
 % WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 % See the License for the specific language governing permissions and
 % limitations under the License.
 %--------------------------------------------------------------------------

function vgg2cenin(vggMatFile) % vgg2cenin('imagenet-matconvnet-vgg-f.mat')
  fprintf('Loading mat file...\n');
  net=load(vggMatFile);
  lc=size(net.layers,2);

  vggMatFile(find(vggMatFile=='.',1,'last'):end)=[]; % remove extension
  
  f=fopen(strcat(vggMatFile,'.cenin'),'w');   % Open an empty file with the same name
  fprintf(f,'CeNiN NEURAL NETWORK FILE');   % Header
  fwrite(f,lc,'int');             % Layer count
  if(isfield(net.meta,'inputSize'))
    s=net.meta.inputSize;
  else
    s=net.meta.inputs.size(1:3);
  end
  for i=1:length(s)
    fwrite(f,s(i),'int'); % Input dimensions (height, width and number of channels (depth))
  end
  for i=1:3
    fwrite(f,net.meta.normalization.averageImage(i),'single');
  end
  for i=1:lc % For each layer
    l=net.layers{i};
    t=l.type;
    s=length(t);
    fwrite(f,s,'int8'); % String length
    fprintf(f,t);     % Layer type (string)

    fprintf('Writing layer %d (%s)...\n',i,l.type);

    if strcmp(t,'conv') % Convolution layers     
      st=l.stride;
      p=l.pad;
      
      % We need 4 padding values for CeNiN (top, bottom, left, right)
      % In vgg format if there are one value, all padding values are
      % the same and if there are two values, these are for top-bottom
      % and left-right paddings.
      if size(st,2)<2 , st(2)=st(1); end
      if size(p,2)<2 , p(2)=p(1); end
      if size(p,2)<3 , p(3:4)=[p(1) p(2)]; end

      % Four padding values
      fwrite(f,p(1),'int8');
      fwrite(f,p(2),'int8');
      fwrite(f,p(3),'int8');
      fwrite(f,p(4),'int8');

      s=size(l.weights{1}); % Dimensions (height, width, number of channels (depth), number of filters)
      for j=1:length(s)
        fwrite(f,s(j),'int');
      end

      % Vertical and horizontal stride values (StrideY and StrideX)
      fwrite(f,st(1),'int8');
      fwrite(f,st(2),'int8');
      
      % Weight values
      % Writing each value one by one takes long time because there are many of them.
      %   for j=1:numel(l.weights{1})
      %     fwrite(f,l.weights{1}(j),'single');
      %   end
      % This is faster:
      fwrite(f,l.weights{1}(:),'single');
      
      % And biases
      %   for j=1:numel(l.weights{2})
      %     fwrite(f,l.weights{2}(j),'single');
      %   end
      fwrite(f,l.weights{2}(:),'single');

    elseif strcmp(t,'relu') % ReLu layers
      % Layer type ('relu') has been written above. There are no extra
      % parameters to be written for this layer..

    elseif strcmp(t,'pool') % Pooling layers
      st=l.stride;
      p=l.pad;
      po=l.pool;
      if size(st,2)<2 , st(2)=st(1); end
      if size(p,2)<2 , p(2)=p(1); end
      if size(p,2)<3 , p(3:4)=[p(1) p(2)]; end
      if size(po,2)<2 , po(2)=po(1); end

      % Four padding values (top, bottom, left, right)
      fwrite(f,p(1),'int8');
      fwrite(f,p(2),'int8');
      fwrite(f,p(3),'int8');
      fwrite(f,p(4),'int8');

      % Vertical and horizontal pooling values (PoolY and PoolX)
      fwrite(f,po(1),'int8');
      fwrite(f,po(2),'int8');

      % Vertical and horizontal stride values (StrideY and StrideX)
      fwrite(f,st(1),'int8');
      fwrite(f,st(2),'int8');

    elseif strcmp(t,'softmax') % SoftMax layer (this is the last layer)
      s=size(net.meta.classes.description,2);
      fwrite(f,s,'int'); % Number of classes
      for j=1:size(net.meta.classes.description,2) % For each class description
        s=size(net.meta.classes.description{j},2);
        fwrite(f,s,'int8'); % String length
        fprintf(f,'%s',net.meta.classes.description{j}); % Class description (string)
      end
    end

  end

  fwrite(f,3,'int8'); % Length of "EOF" as if it is a layer type.
  fprintf(f,'EOF');   % And the "EOF" string itself...
  fclose(f);

end