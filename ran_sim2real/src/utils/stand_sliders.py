import customtkinter as ctk

class JointController(ctk.CTk):
    def __init__(self, angle_range, default):
        super().__init__()

        self.angle_range = angle_range
        self.default = default
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("Joint Controller")
        self.geometry("600x750")
        
        self.slider_widgets = []
        self.checkbox_vars = []
        self.value_labels = []

        self.label_title = ctk.CTkLabel(self, text="Joint Control Panel", font=("Arial", 20, "bold"))
        self.label_title.pack(pady=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=500, height=550)
        self.scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)
        for i in range(12):
            print(i)
            # print(self.angle_range[i][0][0])
            self.create_slider_row(i, self.angle_range[i][0][0] , self.angle_range[i][0][1], self.default[i])

        # self.btn_read = ctk.CTkButton(self, text="Print All Values", command=self.read_values, height=40)
        # self.btn_read.pack(pady=10)

    def create_slider_row(self, index, lower_limit, upper_limit, default):
        

        row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)

        chk_var = ctk.IntVar(value=0)
        self.checkbox_vars.append(chk_var)
        
        checkbox = ctk.CTkCheckBox(
            row_frame, 
            text=f"Joint {index+1}", 
            variable=chk_var, 
            width=80
        )
        checkbox.pack(side="left", padx=10)

        # 2. Slider
        # We use a lambda to pass the specific index of the slider being moved
        slider = ctk.CTkSlider(
            row_frame, 
            from_=lower_limit, 
            to=upper_limit, 
            number_of_steps=(upper_limit - lower_limit) * 2,
            command=lambda v, idx=index: self.on_slider_change(v, idx)
        )
        slider.set(default)
        slider.pack(side="left", fill="x", expand=True, padx=10)
        self.slider_widgets.append(slider)

        # 3. Value Label
        val_label = ctk.CTkLabel(row_frame, text="0.0", width=40)
        val_label.pack(side="right", padx=10)
        self.value_labels.append(val_label)

    def on_slider_change(self, value, index):
        self.value_labels[index].configure(text=f"{value:.1f}")

        is_active = self.checkbox_vars[index].get()

        if is_active:
            for i, other_slider in enumerate(self.slider_widgets):
                if i == index:
                    continue
                
                if self.checkbox_vars[i].get() == 1:
                    other_slider.set(value)
                    self.value_labels[i].configure(text=f"{value:.1f}")


    def read_values(self):
        self.angles = []
        for i, slider in enumerate(self.slider_widgets):
            val = slider.get()
            self.angles.append(val)
        return self.angles
    

ranges = [([[-60.00014031,  60.00014031]]), ([[-60.00014031,  60.00014031]]), ([[-60.00014031,  60.00014031]]), ([[-60.00014031,  60.00014031]]), ([[-90.00021046, 200.00237755]]), ([[-90.00021046, 200.00237755]]), ([[-30.00007015, 260.00251785]]), ([[-30.00007015, 260.00251785]]), ([[-155.99921888,  -48.00011224]]), ([[-155.99921888,  -48.00011224]]), ([[-155.99921888,  -48.00011224]]), ([[-155.99921888,  -48.00011224]])]
print(len(ranges))
if __name__ == "__main__":
    
    a, b, c, d = 0, 45, -100, -120
    controller = JointController(ranges, [0, 0, 0, 0, 46, 46, 57, 57, -86, -86, -86, -86])

    controller.read_values()
    controller.mainloop()       