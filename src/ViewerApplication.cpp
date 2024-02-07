#include "ViewerApplication.hpp"

#include <iostream>
#include <numeric>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/io.hpp>

#include "utils/cameras.hpp"
#include "utils/gltf.hpp"
#include "utils/images.hpp"

#include <stb_image_write.h>
#include <tiny_gltf.h>

void keyCallback(
    GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
    glfwSetWindowShouldClose(window, 1);
  }
}

bool ViewerApplication::loadGltFile(tinygltf::Model &model)
{
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;
  
  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, m_gltfFilePath.string());

  if(! warn.empty())
    std::cerr << "Warn: " << warn.c_str() << "\n";
  
  if(! err.empty())
    std::cerr << "Err: " << err.c_str() << "\n";
  
  if(! ret)
  {
    std::cerr << "Failed to load glTF file" << "\n";

    return false;
  }

  return true;
}

std::vector<GLuint> ViewerApplication::createBufferObjects(const tinygltf::Model model) const
{
  auto buffersSize = model.buffers.size();
  std::vector<GLuint> bufferObjects(buffersSize, 0);

  glGenBuffers(GLsizei(buffersSize), bufferObjects.data());

  for(size_t i = 0; i < buffersSize; i++)
  {
    glBindBuffer(GL_ARRAY_BUFFER, bufferObjects[i]);
    glBufferStorage(GL_ARRAY_BUFFER, model.buffers[i].data.size(), model.buffers[i].data.data(), 0);
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return bufferObjects;
}

std::vector<GLuint> ViewerApplication::createVertexArrayObjects( const tinygltf::Model &model, const std::vector<GLuint> &bufferObjects, std::vector<VaoRange> & meshIndexToVaoRange) const
{
  const GLuint VERTEX_ATTRIB_POSITION_IDX = 0;
  const GLuint VERTEX_ATTRIB_NORMAL_IDX = 1;
  const GLuint VERTEX_ATTRIB_TEXCOORD0_IDX = 2;

  std::vector<GLuint> vertexArrayObjects;

  meshIndexToVaoRange.resize(model.meshes.size());

  for(size_t i = 0; i < model.meshes.size(); i++)
  {
    const auto &mesh = model.meshes[i];
    
    auto vaoRange = meshIndexToVaoRange[i]; 
    vaoRange.begin = GLsizei(vertexArrayObjects.size());
    vaoRange.count = GLsizei(mesh.primitives.size()) ;

    vertexArrayObjects.resize(vertexArrayObjects.size() + mesh.primitives.size());
    
    meshIndexToVaoRange[i] = vaoRange;

    // Create new vertex array objects
    glGenVertexArrays(vaoRange.count, &vertexArrayObjects[vaoRange.begin]);

    for(size_t j = 0; j < mesh.primitives.size(); j++)
    {
      const auto &primitive = mesh.primitives[j];
      glBindVertexArray(vertexArrayObjects[vaoRange.begin + j]);
      // TODO: Fix retarded code duplication
      {
        const auto iterator = primitive.attributes.find("POSITION");
        if(iterator != end(primitive.attributes))
        {
          const auto accessorIdx = (* iterator).second;
          const auto &accressor = model.accessors[accessorIdx];
        
          const auto &bufferView = model.bufferViews[accressor.bufferView];
          const auto bufferIdx = bufferView.buffer;

          // Enable the vertex attrib array corresponding to POSITION
          glEnableVertexAttribArray(VERTEX_ATTRIB_POSITION_IDX);
           assert(GL_ARRAY_BUFFER == bufferView.target);
          // Bind the buffer object to GL_ARRAY_BUFFER
          glBindBuffer(GL_ARRAY_BUFFER, bufferObjects[bufferIdx]);

          // Calculate byte offset for glVertexAttribPointer()
          const auto byteOffset = accressor.byteOffset + bufferView.byteOffset;

          glVertexAttribPointer(VERTEX_ATTRIB_POSITION_IDX, accressor.type, accressor.componentType, GL_FALSE, GLsizei(bufferView.byteStride), (const GLvoid *) byteOffset);
        }
      }

      {
        const auto iterator = primitive.attributes.find("NORMAL");

        if(iterator != end(primitive.attributes))
        {
          const auto accessorIdx = (* iterator).second;
          const auto &accressor = model.accessors[accessorIdx];
        
          const auto &bufferView = model.bufferViews[accressor.bufferView];
          const auto bufferIdx = bufferView.buffer;

          // Enable the vertex attrib array corresponding to NORMAL
          glEnableVertexAttribArray(VERTEX_ATTRIB_NORMAL_IDX);
          assert(GL_ARRAY_BUFFER == bufferView.target);
          // Bind the buffer object to GL_ARRAY_BUFFER
          glBindBuffer(GL_ARRAY_BUFFER, bufferObjects[bufferIdx]);

          // Calculate byte offset for glVertexAttribPointer()
          const auto byteOffset = accressor.byteOffset + bufferView.byteOffset;

          glVertexAttribPointer(VERTEX_ATTRIB_NORMAL_IDX, accressor.type, accressor.componentType, GL_FALSE, GLsizei(bufferView.byteStride), (const GLvoid *) byteOffset);
        }
      }

      {
        const auto iterator = primitive.attributes.find("TEXCOORD_0");

        if(iterator != end(primitive.attributes))
        {
          const auto accessorIdx = (* iterator).second;
          const auto &accressor = model.accessors[accessorIdx];
        
          const auto &bufferView = model.bufferViews[accressor.bufferView];
          const auto bufferIdx = bufferView.buffer;

          // Enable the vertex attrib array corresponding to TEXCOORD_0
          glEnableVertexAttribArray(VERTEX_ATTRIB_TEXCOORD0_IDX);
          assert(GL_ARRAY_BUFFER == bufferView.target);
          // Bind the buffer object to GL_ARRAY_BUFFER
          glBindBuffer(GL_ARRAY_BUFFER, bufferObjects[bufferIdx]);

          // Calculate byte offset for glVertexAttribPointer()
          const auto byteOffset = accressor.byteOffset + bufferView.byteOffset;

          glVertexAttribPointer(VERTEX_ATTRIB_TEXCOORD0_IDX, accressor.type, accressor.componentType, GL_FALSE, GLsizei(bufferView.byteStride), (const GLvoid *) byteOffset);
        }
      }

      if(primitive.indices >= 0)
      {
        const auto accessorIdx = primitive.indices;
        const auto &accressor = model.accessors[accessorIdx];
        
        const auto &bufferView = model.bufferViews[accressor.bufferView];
        const auto bufferIdx = bufferView.buffer;

        const auto bufferObject = bufferObjects[bufferIdx];

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObjects[bufferIdx]);
      }
    }
  }

  glBindVertexArray(0);

  return vertexArrayObjects;
}

std::vector<GLuint> ViewerApplication::createTextureObjects(const tinygltf::Model &model) const
{
  std::vector<GLuint> textureObjects(model.textures.size(), 0);

  tinygltf::Sampler defaultSampler;
  
  defaultSampler.minFilter = GL_LINEAR;
  defaultSampler.magFilter = GL_LINEAR;
  defaultSampler.wrapS = GL_REPEAT;
  defaultSampler.wrapR = GL_REPEAT;
  defaultSampler.wrapT = GL_REPEAT;

  glActiveTexture(GL_TEXTURE0);

  glGenTextures(GLsizei(model.textures.size()), textureObjects.data());

  
  for(size_t i = 0; i < model.textures.size(); i++)
  {
    const auto &texture = model.textures[i];
    assert(texture.source >= 0);

    const auto &sampler = texture.sampler >= 0 ? model.samplers[texture.sampler] : defaultSampler;
    const auto &image = model.images[texture.source];

    glBindTexture(GL_TEXTURE_2D, textureObjects[i]);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, image.pixel_type, image.image.data());
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sampler.minFilter != -1 ? sampler.minFilter : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sampler.magFilter != -1 ? sampler.magFilter : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, sampler.wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, sampler.wrapR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, sampler.wrapT);

    if(sampler.minFilter == GL_NEAREST_MIPMAP_NEAREST || sampler.minFilter == GL_NEAREST_MIPMAP_LINEAR ||
       sampler.minFilter == GL_LINEAR_MIPMAP_NEAREST  || sampler.minFilter == GL_LINEAR_MIPMAP_LINEAR)
    {
      glGenerateMipmap(GL_TEXTURE_2D);
    }
  }

  glBindTexture(GL_TEXTURE_2D, 0);

  return textureObjects;
}


int ViewerApplication::run()
{
  // Loader shaders
  const auto glslProgram =
      compileProgram({m_ShadersRootPath / m_vertexShader,
          m_ShadersRootPath / m_fragmentShader});

  const auto modelViewProjMatrixLocation = glGetUniformLocation(glslProgram.glId(), "uModelViewProjMatrix");
  const auto modelViewMatrixLocation = glGetUniformLocation(glslProgram.glId(), "uModelViewMatrix");
  const auto normalMatrixLocation = glGetUniformLocation(glslProgram.glId(), "uNormalMatrix");
  const auto lightDirectionLocation = glGetUniformLocation(glslProgram.glId(), "uLightDirection");
  const auto lightIntensityLocation = glGetUniformLocation(glslProgram.glId(), "uLightIntensity");
  const auto baseColorTextureLocation = glGetUniformLocation(glslProgram.glId(), "uBaseColorTexture");
  const auto baseColorFactorLocation = glGetUniformLocation(glslProgram.glId(), "uBaseColorFactor");
  const auto metallicFactorLocation = glGetUniformLocation(glslProgram.glId(), "uMetallicFactor");
  const auto roughnessFactorLocation = glGetUniformLocation(glslProgram.glId(), "uRoughnessFactor");
  const auto metallicRoughnessTexLocation = glGetUniformLocation(glslProgram.glId(), "uMetallicRoughnessTexture");
  const auto emissiveFacorLocation = glGetUniformLocation(glslProgram.glId(), "uEmissiveFactor");
  const auto emissiveTextureLocation = glGetUniformLocation(glslProgram.glId(), "uEmissiveTexture");
  const auto occlusionStrengthLocation = glGetUniformLocation(glslProgram.glId(), "uOcclusionStrength");
  const auto occlusionTextureLocation = glGetUniformLocation(glslProgram.glId(), "uOcclusionTexture");
  const auto appyOcclusionLocation = glGetUniformLocation(glslProgram.glId(), "uApplyOcclusion");


  // Load glTF model
  tinygltf::Model model;

  if(! loadGltFile(model))
    return -1;

  glm::vec3 bboxMin, bboxMax;
  
  // Compute bounding box of a scene
  computeSceneBounds(model, bboxMin, bboxMax);

  // Build projection matrix
  const auto diag = bboxMax - bboxMin; 
  auto maxDistance = glm::length(diag);

  const auto projMatrix =
      glm::perspective(70.f, float(m_nWindowWidth) / m_nWindowHeight,
          0.001f * maxDistance, 1.5f * maxDistance);
          
  std::unique_ptr<CameraController> cameraController = 
    std::make_unique<TrackballCameraController>(m_GLFWHandle.window(), 0.5f * maxDistance);
  
  if (m_hasUserCamera) 
  {
    cameraController->setCamera(m_userCamera);
  } 
  else 
  {
    const auto center = 0.5f * (bboxMin + bboxMax);
    const auto up = glm::vec3(0, 1, 0);
    const auto eye = diag.z > 0 ? center + diag : center + 2.f * glm::cross(diag, up);

    cameraController->setCamera(Camera{eye, center, up});
  }

  // TODO Creation of Buffer Objects

  glm::vec3 lightDirection(1, 1, 1), lightIntensity(1, 1, 1);
  bool lightFromCamera = false;
  bool applyOcclusion = true;

  const auto textureObjects = createTextureObjects(model);

  float white[] = {1, 1, 1, 1};
  GLuint whiteTexture = 0;

  glGenTextures(1, &whiteTexture);
  glBindTexture(GL_TEXTURE_2D, whiteTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_FLOAT, white);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
  glBindTexture(GL_TEXTURE_2D, 0);


  const auto bufferObjects = createBufferObjects(model);

  // TODO Creation of Vertex Array Objects
  std::vector<VaoRange> meshToVertexArrays;
  const auto vertexArrayObjects = createVertexArrayObjects(model, bufferObjects, meshToVertexArrays);

  // Setup OpenGL state for rendering
  glEnable(GL_DEPTH_TEST);
  glslProgram.use();

  const auto bindMaterial = [&](const auto materialIndex)
  {
    if(materialIndex >= 0)
    {
      const auto &material = model.materials[materialIndex];
      const auto &pbrMetallicRoughness = material.pbrMetallicRoughness;

      if(baseColorFactorLocation >= 0)
      {
        glUniform4f(baseColorFactorLocation,
          (float)pbrMetallicRoughness.baseColorFactor[0],
          (float)pbrMetallicRoughness.baseColorFactor[1],
          (float)pbrMetallicRoughness.baseColorFactor[2],
          (float)pbrMetallicRoughness.baseColorFactor[3]);
      }
      
      if(baseColorTextureLocation >= 0)
      {
        auto textureObject = whiteTexture;

        if(pbrMetallicRoughness.baseColorTexture.index >= 0)
        {
          const auto &texture = model.textures[pbrMetallicRoughness.baseColorTexture.index];

          if(texture.source >= 0)
            textureObject = textureObjects[texture.source];
        }

        if(baseColorFactorLocation)
          glUniform4f(baseColorFactorLocation, 1, 1, 1, 1);

        if(baseColorTextureLocation >= 0)
        {
          glActiveTexture(GL_TEXTURE0);
          glBindTexture(GL_TEXTURE_2D, textureObject);
        }

        glUniform1i(baseColorTextureLocation, 0);
      }

      if(metallicFactorLocation >= 0)
        glUniform1f(metallicFactorLocation, (float)pbrMetallicRoughness.metallicFactor);

      if(roughnessFactorLocation >= 0)
        glUniform1f(roughnessFactorLocation, (float)pbrMetallicRoughness.roughnessFactor);
      
      if(metallicRoughnessTexLocation >= 0)
      {
        auto textureObject = 0u;

        if(pbrMetallicRoughness.metallicRoughnessTexture.index >= 0)
        {
          const auto &texture = model.textures[pbrMetallicRoughness.metallicRoughnessTexture.index];
        
          if(texture.source >= 0)
            textureObject = textureObjects[texture.source];       
        }

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textureObject);
        glUniform1i(metallicRoughnessTexLocation, 1);
      }

      if(emissiveFacorLocation >= 0)
        glUniform3f(emissiveFacorLocation, (float)material.emissiveFactor[0], (float)material.emissiveFactor[1], (float)material.emissiveFactor[2]);
      
      if(emissiveTextureLocation >= 0)
      {
        auto textureObject = 0u;

        if(material.emissiveTexture.index >= 0)
        { 
          const auto &texture = model.textures[material.emissiveTexture.index];

          if(texture.source >= 0)
            textureObject = textureObjects[texture.source];
        }

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, textureObject);
        glUniform1i(emissiveTextureLocation, 2);
      }

      if(occlusionStrengthLocation >= 0)
        glUniform1f(occlusionStrengthLocation, (float)material.occlusionTexture.strength);

      if(occlusionTextureLocation >= 0)
      {
        auto &textureObject = whiteTexture;

        if(material.occlusionTexture.index >= 0)
        {
          const auto &texture = model.textures[material.occlusionTexture.index];

          if(texture.source >= 0)
            textureObject = textureObjects[texture.source];
        }

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, textureObject);
        glUniform1i(occlusionTextureLocation, 3);
      }
    }

    else
    {
      if (baseColorTextureLocation >= 0) 
      {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, whiteTexture);
        glUniform1i(baseColorTextureLocation, 0);
      }

      if(metallicFactorLocation >= 0)
        glUniform1f(metallicFactorLocation, 1.f);
      
      if(roughnessFactorLocation >= 0)
        glUniform1f(roughnessFactorLocation, 1.f);
      
      if(metallicRoughnessTexLocation >= 0)
      {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUniform1i(metallicRoughnessTexLocation, 1);
      }

      if(emissiveFacorLocation >= 0)
        glUniform3f(emissiveFacorLocation, 0.f, 0.f, 0.f);

      if(emissiveTextureLocation >= 0)
      {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUniform1i(emissiveTextureLocation, 2);
      }

      if(occlusionStrengthLocation >= 0)
        glUniform1f(occlusionStrengthLocation, 0.f);
      
      if(occlusionTextureLocation >= 0)
      {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUniform1i(occlusionTextureLocation, 3);
      }
    }
  };

  // Lambda function to draw the scene
  const auto drawScene = [&](const Camera &camera) 
  {
    glViewport(0, 0, m_nWindowWidth, m_nWindowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto viewMatrix = camera.getViewMatrix();

    if(lightDirectionLocation >= 0)
    {
      if(lightFromCamera)
        glUniform3f(lightDirectionLocation, 0, 0, 1);
      
      else
      {
        const auto lightDirectionInViewSpace = 
          glm::normalize(glm::vec3(viewMatrix * glm::vec4(lightDirection, 0.)));
      
        glUniform3f(lightDirectionLocation, lightDirectionInViewSpace[0], 
          lightDirectionInViewSpace[1], lightDirectionInViewSpace[2]);
      }
    }

    if(lightIntensityLocation >= 0)
      glUniform3f(lightIntensityLocation, lightIntensity[0], lightIntensity[1], lightIntensity[2]);
    
    if(appyOcclusionLocation >= 0)
      glUniform1i(appyOcclusionLocation, applyOcclusion);

    // The recursive function that should draw a node
    // We use a std::function because a simple lambda cannot be recursive
    const std::function<void(int, const glm::mat4 &)> drawNode =
        [&](int nodeIdx, const glm::mat4 &parentMatrix)
        {
          // TODO The drawNode function
          const auto node = model.nodes[nodeIdx];
          
          const glm::mat4 modelMatrix = getLocalToWorldMatrix(node, parentMatrix);

          if(node.mesh >= 0)
          {
            const auto mvMatrix = viewMatrix * modelMatrix;
            const auto mvpMatrix = projMatrix * mvMatrix;
            const auto normalMatrix = glm::transpose(glm::inverse(mvMatrix));

            glUniformMatrix4fv(modelViewMatrixLocation, 1, GL_FALSE, glm::value_ptr(mvMatrix));
            glUniformMatrix4fv(modelViewProjMatrixLocation, 1, GL_FALSE, glm::value_ptr(mvpMatrix));
            glUniformMatrix4fv(normalMatrixLocation, 1, GL_FALSE, glm::value_ptr(normalMatrix));

            const auto &mesh = model.meshes[node.mesh];
            const auto &vaoRange = meshToVertexArrays[node.mesh];

            for(size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++)
            {
              const auto &primitive = mesh.primitives[primIdx];
              const auto prim_vao = vertexArrayObjects[vaoRange.begin + primIdx];

              bindMaterial(primitive.material);

              glBindVertexArray(prim_vao);

              if(primitive.indices >= 0)
              {
                const auto &accessor = model.accessors[primitive.indices];
                const auto &bufferView = model.bufferViews[accessor.bufferView];
                const auto byteOffset = accessor.byteOffset + bufferView.byteOffset;
                
                glDrawElements(primitive.mode, GLsizei(accessor.count), accessor.componentType, (const GLvoid *)byteOffset);
              }
              else
              {
                const auto accessorIdx = (*begin(primitive.attributes)).second;
                const auto &accessor = model.accessors[accessorIdx];

                glDrawArrays(primitive.mode, 0, GLsizei(accessor.count));
              }

              for(const auto childNodeIdx : node.children)
                drawNode(childNodeIdx, modelMatrix);
            }
          }
        };

    // Draw the scene referenced by gltf file
    if (model.defaultScene >= 0) 
    {
      // TODO Draw all nodes
      for(const auto nodeIdx : model.scenes[model.defaultScene].nodes)
        drawNode(nodeIdx, glm::mat4(1));
    }
  };

  if(! m_OutputPath.empty())
  {
    const auto numComponents = 3;
    std::vector<unsigned char> pixels(m_nWindowWidth * m_nWindowHeight * numComponents);
    renderToImage(m_nWindowWidth, m_nWindowHeight, numComponents, pixels.data(), [&]() 
      {
        const auto camera = cameraController->getCamera();
        drawScene(camera);
      });

      flipImageYAxis(m_nWindowWidth, m_nWindowHeight, numComponents, pixels.data());

      const auto strPath = m_OutputPath.string();
      stbi_write_png(strPath.c_str(), m_nWindowWidth, m_nWindowHeight, numComponents, pixels.data(), 0);

      return 0;
  }

  // Loop until the user closes the window
  for (auto iterationCount = 0u; !m_GLFWHandle.shouldClose();
       ++iterationCount) {
    const auto seconds = glfwGetTime();

    const auto camera = cameraController->getCamera();
    drawScene(camera);

    // GUI code:
    imguiNewFrame();

    {
      ImGui::Begin("GUI");
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
          1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("eye: %.3f %.3f %.3f", camera.eye().x, camera.eye().y,
            camera.eye().z);
        ImGui::Text("center: %.3f %.3f %.3f", camera.center().x,
            camera.center().y, camera.center().z);
        ImGui::Text(
            "up: %.3f %.3f %.3f", camera.up().x, camera.up().y, camera.up().z);

        ImGui::Text("front: %.3f %.3f %.3f", camera.front().x, camera.front().y,
            camera.front().z);
        ImGui::Text("left: %.3f %.3f %.3f", camera.left().x, camera.left().y,
            camera.left().z);

        if (ImGui::Button("CLI camera args to clipboard")) {
          std::stringstream ss;
          ss << "--lookat " << camera.eye().x << "," << camera.eye().y << ","
             << camera.eye().z << "," << camera.center().x << ","
             << camera.center().y << "," << camera.center().z << ","
             << camera.up().x << "," << camera.up().y << "," << camera.up().z;
          const auto str = ss.str();
          glfwSetClipboardString(m_GLFWHandle.window(), str.c_str());
        }

        static int cameraControllerType = 0;
        const auto cameraControllerTypeChanged = 
          ImGui::RadioButton("Trackball Camera", &cameraControllerType, 0) ||
          ImGui::RadioButton("First Person Camera", &cameraControllerType, 1);

          if(cameraControllerTypeChanged)
          {
            const auto currentCamera = cameraController->getCamera();

            if(cameraControllerType == 0)
              cameraController = std::make_unique<TrackballCameraController>(m_GLFWHandle.window(), 0.5f * maxDistance);

            else
              cameraController = std::make_unique<FirstPersonCameraController>(m_GLFWHandle.window(), 0.5f * maxDistance);

            cameraController->setCamera(currentCamera);
          }
        
        if(ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
        {
          static float lightTheta = .0f, lightPhi = .0f;

          if(ImGui::SliderFloat("Theta", &lightTheta, 0, glm::pi<float>()) || 
             ImGui::SliderFloat("Phi", &lightPhi, 0, 2.f * glm::pi<float>()))
          {
            auto lightDirectionX = glm::sin(lightTheta) * glm::cos(lightPhi);
            auto lightDirectionY = glm::cos(lightTheta);
            auto lightDirectionZ = glm::sin(lightTheta) * glm::sin(lightPhi);
            
            auto lightDirection = glm::vec3(lightDirectionX, lightDirectionY, lightDirectionZ);
          }

          static glm::vec3 lightColor(1.f, 1.f, 1.f);
          static float lightIntensityFactor = 1.f;

          if(ImGui::ColorEdit3("Light Color", (float *)&lightColor) || 
             ImGui::InputFloat("Light Intensity", &lightIntensityFactor))
          {
            lightIntensity = lightColor * lightIntensityFactor;
          }
        }
        ImGui::Checkbox("Light From Camera", &lightFromCamera);
        ImGui::Checkbox("Ambient Occlusion", &applyOcclusion);
      }
      ImGui::End();
    }

    imguiRenderFrame();

    glfwPollEvents(); // Poll for and process events

    auto ellapsedTime = glfwGetTime() - seconds;
    auto guiHasFocus =
        ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;
    if (!guiHasFocus) {
      cameraController->update(float(ellapsedTime));
    }

    m_GLFWHandle.swapBuffers(); // Swap front and back buffers
  }

  // TODO clean up allocated GL data

  return 0;
}

ViewerApplication::ViewerApplication(const fs::path &appPath, uint32_t width,
    uint32_t height, const fs::path &gltfFile,
    const std::vector<float> &lookatArgs, const std::string &vertexShader,
    const std::string &fragmentShader, const fs::path &output) :
    m_nWindowWidth(width),
    m_nWindowHeight(height),
    m_AppPath{appPath},
    m_AppName{m_AppPath.stem().string()},
    m_ImGuiIniFilename{m_AppName + ".imgui.ini"},
    m_ShadersRootPath{m_AppPath.parent_path() / "shaders"},
    m_gltfFilePath{gltfFile},
    m_OutputPath{output}
{
  if (!lookatArgs.empty()) {
    m_hasUserCamera = true;
    m_userCamera =
        Camera{glm::vec3(lookatArgs[0], lookatArgs[1], lookatArgs[2]),
            glm::vec3(lookatArgs[3], lookatArgs[4], lookatArgs[5]),
            glm::vec3(lookatArgs[6], lookatArgs[7], lookatArgs[8])};
  }

  if (!vertexShader.empty()) {
    m_vertexShader = vertexShader;
  }

  if (!fragmentShader.empty()) {
    m_fragmentShader = fragmentShader;
  }

  ImGui::GetIO().IniFilename =
      m_ImGuiIniFilename.c_str(); // At exit, ImGUI will store its windows
                                  // positions in this file

  glfwSetKeyCallback(m_GLFWHandle.window(), keyCallback);

  printGLVersion();
}
