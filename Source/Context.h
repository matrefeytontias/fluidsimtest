#pragma once

#include <Empty/Context.hpp>
#include <Empty/gl/Framebuffer.h>
#include <Empty/gl/Texture.h>
#include <Empty/gl/Renderbuffer.h>
#include <Empty/utils/macros.h>
#include <Empty/utils/noncopyable.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

struct Context : public Empty::Context, Empty::utils::noncopyable
{
    ~Context() override
    {
        terminate();
    }

    static Context& get() { return _instance; }

    bool init(const char* title, int w, int h)
    {
        ASSERT(!_init);
        /// Setup window
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
        {
            TRACE("Couldn't initialize GLFW");
            return false;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
        window = glfwCreateWindow(w, h, title, NULL, NULL);
        if (!window)
        {
            TRACE("Couldn't create window");
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);
        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        glfwSwapInterval(0); // no v-sync, live on the edge

        /// Setup ImGui binding
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
        ImGui_ImplOpenGL3_Init("#version 450");
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);

        /// Setup style
        ImGui::StyleColorsDark();
        // ImGui::StyleColorsClassic();

        setViewport(w, h);
        frameWidth = w;
        frameHeight = h;

        Empty::gl::Framebuffer::initDefaultFramebuffer();

        _init = true;
        return true;
    }

    void newFrame() const
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void swap() const override
    {
        using namespace Empty::gl;

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwPollEvents();
        glfwSwapBuffers(window);
        Framebuffer::dflt->clearAttachment<FramebufferAttachment::Color>(0, Empty::math::vec4::zero);
    }

    void terminate()
    {
        if (_init)
        {
            ImGui_ImplGlfw_Shutdown();
            ImGui_ImplOpenGL3_Shutdown();
            ImGui::DestroyContext();
            glfwTerminate();
            _init = false;
        }
    }

    int frameWidth, frameHeight;
    GLFWwindow* window;

private:
    Context() : Empty::Context(), Empty::utils::noncopyable(), frameWidth(0), frameHeight(0), window(nullptr), _init(false) { }
    bool _init;
    static Context _instance;

    // Callbacks
    static void glfw_error_callback(int error, const char* description)
    {
        TRACE("Error " << error << " : " << description);
    }

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        Context& context = Context::get();
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    };

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
    {
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    }
};
