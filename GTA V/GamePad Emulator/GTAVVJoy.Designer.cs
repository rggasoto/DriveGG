namespace VJoyDemo
{
    partial class GTAVVJoy
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(GTAVVJoy));
            this.trkXAxis = new System.Windows.Forms.TrackBar();
            this.lblXAxis = new System.Windows.Forms.Label();
            this.lblYAxis = new System.Windows.Forms.Label();
            this.trkYAxis = new System.Windows.Forms.TrackBar();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.cbEBrake = new System.Windows.Forms.CheckBox();
            this.lblConnected = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.trkXAxis)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trkYAxis)).BeginInit();
            this.SuspendLayout();
            // 
            // trkXAxis
            // 
            this.trkXAxis.Dock = System.Windows.Forms.DockStyle.Top;
            this.trkXAxis.LargeChange = 1;
            this.trkXAxis.Location = new System.Drawing.Point(0, 0);
            this.trkXAxis.Maximum = 511;
            this.trkXAxis.Name = "trkXAxis";
            this.trkXAxis.Size = new System.Drawing.Size(885, 45);
            this.trkXAxis.TabIndex = 2;
            this.trkXAxis.TickFrequency = 8;
            this.trkXAxis.TickStyle = System.Windows.Forms.TickStyle.Both;
            this.trkXAxis.Value = 255;
            this.trkXAxis.Scroll += new System.EventHandler(this.trkXAxis_Scroll);
            this.trkXAxis.ValueChanged += new System.EventHandler(this.trkXAxis_Scroll);
            // 
            // lblXAxis
            // 
            this.lblXAxis.AutoSize = true;
            this.lblXAxis.Dock = System.Windows.Forms.DockStyle.Top;
            this.lblXAxis.Location = new System.Drawing.Point(0, 45);
            this.lblXAxis.Name = "lblXAxis";
            this.lblXAxis.Size = new System.Drawing.Size(66, 13);
            this.lblXAxis.TabIndex = 3;
            this.lblXAxis.Text = "Acceleration";
            // 
            // lblYAxis
            // 
            this.lblYAxis.AutoSize = true;
            this.lblYAxis.Dock = System.Windows.Forms.DockStyle.Top;
            this.lblYAxis.Location = new System.Drawing.Point(0, 103);
            this.lblYAxis.Name = "lblYAxis";
            this.lblYAxis.Size = new System.Drawing.Size(46, 13);
            this.lblYAxis.TabIndex = 7;
            this.lblYAxis.Text = "Steering";
            // 
            // trkYAxis
            // 
            this.trkYAxis.Dock = System.Windows.Forms.DockStyle.Top;
            this.trkYAxis.Location = new System.Drawing.Point(0, 58);
            this.trkYAxis.Maximum = 32767;
            this.trkYAxis.Name = "trkYAxis";
            this.trkYAxis.Size = new System.Drawing.Size(885, 45);
            this.trkYAxis.TabIndex = 6;
            this.trkYAxis.TickFrequency = 1024;
            this.trkYAxis.TickStyle = System.Windows.Forms.TickStyle.Both;
            this.trkYAxis.Value = 16384;
            this.trkYAxis.Scroll += new System.EventHandler(this.trkYAxis_Scroll);
            this.trkYAxis.ValueChanged += new System.EventHandler(this.trkYAxis_Scroll);
            // 
            // timer1
            // 
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // cbEBrake
            // 
            this.cbEBrake.Appearance = System.Windows.Forms.Appearance.Button;
            this.cbEBrake.AutoSize = true;
            this.cbEBrake.BackColor = System.Drawing.Color.Red;
            this.cbEBrake.BackgroundImageLayout = System.Windows.Forms.ImageLayout.None;
            this.cbEBrake.Dock = System.Windows.Forms.DockStyle.Fill;
            this.cbEBrake.FlatAppearance.BorderSize = 0;
            this.cbEBrake.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.cbEBrake.Location = new System.Drawing.Point(0, 116);
            this.cbEBrake.Name = "cbEBrake";
            this.cbEBrake.Size = new System.Drawing.Size(885, 87);
            this.cbEBrake.TabIndex = 9;
            this.cbEBrake.Tag = "1";
            this.cbEBrake.Text = "e-brake";
            this.cbEBrake.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.cbEBrake.UseVisualStyleBackColor = false;
            this.cbEBrake.CheckedChanged += new System.EventHandler(this.Button_CheckedChanged);
            // 
            // lblConnected
            // 
            this.lblConnected.Dock = System.Windows.Forms.DockStyle.Right;
            this.lblConnected.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.lblConnected.ForeColor = System.Drawing.Color.DarkRed;
            this.lblConnected.Location = new System.Drawing.Point(812, 116);
            this.lblConnected.Name = "lblConnected";
            this.lblConnected.Size = new System.Drawing.Size(73, 87);
            this.lblConnected.TabIndex = 10;
            this.lblConnected.Text = "Disconnected";
            this.lblConnected.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // GTAVVJoy
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(885, 203);
            this.Controls.Add(this.lblConnected);
            this.Controls.Add(this.cbEBrake);
            this.Controls.Add(this.lblYAxis);
            this.Controls.Add(this.trkYAxis);
            this.Controls.Add(this.lblXAxis);
            this.Controls.Add(this.trkXAxis);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "GTAVVJoy";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "GTAV Controller";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.trkXAxis)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trkYAxis)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.TrackBar trkXAxis;
        private System.Windows.Forms.Label lblXAxis;
        private System.Windows.Forms.Label lblYAxis;
        private System.Windows.Forms.TrackBar trkYAxis;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.CheckBox cbEBrake;
        private System.Windows.Forms.Label lblConnected;
    }
}

