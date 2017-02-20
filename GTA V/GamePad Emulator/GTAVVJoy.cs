using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using vJoyInterfaceWrap;

namespace VJoyDemo
{
    public partial class GTAVVJoy : Form
    {
        const string DEFAULT_SERVER = "localhost";
        const int DEFAULT_PORT = 804;
        Socket clientSocket;
        IPHostEntry hostInfo;
        IPAddress serverAddr;
        double acc;
        double steering;
        double ebrake;
        static public vJoy joystick;
        static public vJoy.JoystickState iReport;
        static public uint id = 1;

        public GTAVVJoy()
        {
            InitializeComponent();

        }

        public void StartListening()
        {
            // Data buffer for incoming data.  
            byte[] bytes = new Byte[1024];

            // Establish the local endpoint for the socket.  
            // Dns.GetHostName returns the name of the   
            // host running the application.  
            hostInfo = Dns.GetHostEntry(DEFAULT_SERVER);
            serverAddr = hostInfo.AddressList[1];
            IPEndPoint localEndPoint = new IPEndPoint(serverAddr, DEFAULT_PORT);

            // Create a TCP/IP socket.  
            Socket listener = new Socket(AddressFamily.InterNetwork,
                SocketType.Stream, ProtocolType.Tcp);

            // Bind the socket to the local endpoint and   
            // listen for incoming connections.  
            try
            {
                listener.Bind(localEndPoint);
                listener.Listen(10);

                // Start listening for connections.  
                while (true)
                {
                    Console.WriteLine("Waiting for a connection...");
                    string data = null;
                    // Program is suspended while waiting for an incoming connection.  
                    Socket handler = listener.Accept();
                    if (this.lblConnected.InvokeRequired)
                    {
                        this.lblConnected.BeginInvoke((MethodInvoker)delegate ()
                        {
                            this.lblConnected.Text = "Connected";
                            this.lblConnected.ForeColor = Color.DarkGreen;
                        });
                    }
                    else
                    {
                        this.lblConnected.Text = "Connected";
                        this.lblConnected.ForeColor = Color.DarkGreen;
                    }
                    data = null;
                    try
                    {
                        // An incoming connection needs to be processed.  
                        while (true)
                        {
                            bytes = new byte[1024];

                            int bytesRec = handler.Receive(bytes);
                            data += Encoding.ASCII.GetString(bytes, 0, bytesRec);
                            if (data.IndexOf("<EOT>") > -1)
                            {
                                string[] commands = data.Split(';');
                                Console.WriteLine(data);
                                this.acc = Math.Min(this.trkXAxis.Maximum,Math.Max(this.trkXAxis.Minimum, Double.Parse(commands[0])));
                                this.steering = Math.Min(this.trkYAxis.Maximum, Math.Max(this.trkYAxis.Minimum, Double.Parse(commands[1])/255*(this.trkYAxis.Maximum - (this.trkYAxis.Minimum)))); 
                                this.ebrake = double.Parse(commands[2]);
                                //joystick.SetAxis((int)((acc * 32767) / 511), id, HID_USAGES.HID_USAGE_Y);
                                //joystick.SetAxis((int)((steering* 32767) / 511), id, HID_USAGES.HID_USAGE_X);
                                //joystick.SetBtn(ebrake !=0, id, 1);
                                
                                updateScreen();
                                data = null;
                            }
                        }
                    }catch (Exception exc)
                    {
                        Console.WriteLine("Disconnected");
                        if (this.cbEBrake.InvokeRequired)
                        {
                            this.cbEBrake.BeginInvoke((MethodInvoker)delegate ()
                            {
                                this.ebrake = 1;
                                this.cbEBrake.Checked = true;
                                Button_CheckedChanged(this.cbEBrake, new EventArgs());
                            });
                        }
                        else
                        {
                            this.ebrake = 1;
                            this.cbEBrake.Checked = true;
                            Button_CheckedChanged(this.cbEBrake, new EventArgs());
                        }
                        if (this.lblConnected.InvokeRequired)
                        {
                            this.lblConnected.BeginInvoke((MethodInvoker)delegate ()
                            {
                                this.lblConnected.Text = "Disconnected";
                                this.lblConnected.ForeColor = Color.DarkRed;
                            });
                        }
                        else
                        {
                            this.lblConnected.Text = "Disconnected";
                            this.lblConnected.ForeColor = Color.DarkRed;
                        }
                    }                   
                    handler.Close();
                }

            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }

            Console.WriteLine("\nPress ENTER to continue...");
            Console.Read();

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            CreateButtons();

            joystick = new vJoy();
            iReport = new vJoy.JoystickState();


            // Get the driver attributes (Vendor ID, Product ID, Version Number)
            if (!joystick.vJoyEnabled())
            {
                Console.WriteLine("vJoy driver not enabled: Failed Getting vJoy attributes.\n");
                return;
            }
            else
                Console.WriteLine("Vendor: {0}\nProduct :{1}\nVersion Number:{2}\n", joystick.GetvJoyManufacturerString(), joystick.GetvJoyProductString(), joystick.GetvJoySerialNumberString());

            // Get the state of the requested device
            VjdStat status = joystick.GetVJDStatus(id);
            switch (status)
            {
                case VjdStat.VJD_STAT_OWN:
                    Console.WriteLine("vJoy Device {0} is already owned by this feeder\n", id);
                    break;
                case VjdStat.VJD_STAT_FREE:
                    Console.WriteLine("vJoy Device {0} is free\n", id);
                    break;
                case VjdStat.VJD_STAT_BUSY:
                    Console.WriteLine("vJoy Device {0} is already owned by another feeder\nCannot continue\n", id);
                    return;
                case VjdStat.VJD_STAT_MISS:
                    Console.WriteLine("vJoy Device {0} is not installed or disabled\nCannot continue\n", id);
                    return;
                default:
                    Console.WriteLine("vJoy Device {0} general error\nCannot continue\n", id);
                    return;
            };

            // Check which axes are supported
            bool AxisX = joystick.GetVJDAxisExist(id, HID_USAGES.HID_USAGE_X);
            bool AxisY = joystick.GetVJDAxisExist(id, HID_USAGES.HID_USAGE_Y);
            bool AxisZ = joystick.GetVJDAxisExist(id, HID_USAGES.HID_USAGE_Z);
            bool AxisRX = joystick.GetVJDAxisExist(id, HID_USAGES.HID_USAGE_RX);
            bool AxisRZ = joystick.GetVJDAxisExist(id, HID_USAGES.HID_USAGE_RZ);
            // Get the number of buttons and POV Hat switchessupported by this vJoy device
            int nButtons = joystick.GetVJDButtonNumber(id);
            int ContPovNumber = joystick.GetVJDContPovNumber(id);
            int DiscPovNumber = joystick.GetVJDDiscPovNumber(id);

            // Print results
            Console.WriteLine("\nvJoy Device {0} capabilities:\n", id);
            Console.WriteLine("Numner of buttons\t\t{0}\n", nButtons);
            Console.WriteLine("Numner of Continuous POVs\t{0}\n", ContPovNumber);
            Console.WriteLine("Numner of Descrete POVs\t\t{0}\n", DiscPovNumber);
            Console.WriteLine("Axis X\t\t{0}\n", AxisX ? "Yes" : "No");
            Console.WriteLine("Axis Y\t\t{0}\n", AxisX ? "Yes" : "No");
            Console.WriteLine("Axis Z\t\t{0}\n", AxisX ? "Yes" : "No");
            Console.WriteLine("Axis Rx\t\t{0}\n", AxisRX ? "Yes" : "No");
            Console.WriteLine("Axis Rz\t\t{0}\n", AxisRZ ? "Yes" : "No");

            // Test if DLL matches the driver
            UInt32 DllVer = 0, DrvVer = 0;
            bool match = joystick.DriverMatch(ref DllVer, ref DrvVer);
            if (match)
                Console.WriteLine("Version of Driver Matches DLL Version ({0:X})\n", DllVer);
            else
                Console.WriteLine("Version of Driver ({0:X}) does NOT match DLL Version ({1:X})\n", DrvVer, DllVer);


            // Acquire the target
            if ((status == VjdStat.VJD_STAT_OWN) || ((status == VjdStat.VJD_STAT_FREE) && (!joystick.AcquireVJD(id))))
            {
                Console.WriteLine("Failed to acquire vJoy device number {0}.\n", id);
                return;
            }
            else
                Console.WriteLine("Acquired: vJoy device number {0}.\n", id);

            Thread t = new Thread(new ThreadStart(this.StartListening));
            t.IsBackground = true;
            t.Start();
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            
        }
        private void updateScreen()
        {
            if (this.cbEBrake.InvokeRequired)
            {
                this.cbEBrake.BeginInvoke((MethodInvoker)delegate ()
                {
                    this.cbEBrake.Checked = this.ebrake != 0;
                    Button_CheckedChanged(this.cbEBrake, new EventArgs());
                });
            }
            else
            {
                this.cbEBrake.Checked = this.ebrake != 0;
                Button_CheckedChanged(this.cbEBrake, new EventArgs());
            }
            if (this.trkXAxis.InvokeRequired)
            {
                this.cbEBrake.BeginInvoke((MethodInvoker)delegate ()
                {
                    trkXAxis.Value = (int)this.acc;
                    trkXAxis_Scroll(this, new EventArgs());
                });
            }
            else
            {
                trkXAxis.Value = (int)this.acc;
                trkXAxis_Scroll(this, new EventArgs());
            }

            if (this.trkYAxis.InvokeRequired)
            {
                this.cbEBrake.BeginInvoke((MethodInvoker)delegate () { trkYAxis.Value = (int)this.steering;
                    trkYAxis_Scroll(this, new EventArgs());
                });
            }
            else { 
                trkYAxis.Value = (int)this.steering;
                trkYAxis_Scroll(this, new EventArgs());
            }
        }
        private void Button_CheckedChanged(object sender, EventArgs e)
        {            
            CheckBox chk = (CheckBox)sender;

            joystick.SetBtn(chk.Checked, id, uint.Parse(chk.Tag.ToString()));
            if (chk.Checked)
            {
                chk.BackColor = Color.DarkRed;
            }
            else
            {
                chk.BackColor = Color.Red;
            }

            
        }
        

        private void CreateButtons()
        {
            //m_button = new CheckBox[32];

            //for (int y = 0; y < 4; y++)
            //{
            //    for (int x = 0; x < 8; x++)
            //    {
            //        int id = y * 8 + x;
            //        Size size = new Size(panel1.Width / 8, panel1.Height / 4);

            //        m_button[id] = new CheckBox();
            //        m_button[id].Tag = id;

            //        m_button[id].Location = new Point(x * size.Width, y * size.Height);
            //        m_button[id].Size = size;

            //        m_button[id].Appearance = Appearance.Button;

            //        m_button[id].Text = id.ToString();

            //        m_button[id].CheckedChanged += new EventHandler(Button_CheckedChanged);

            //        panel1.Controls.Add(m_button[id]);
            //    }
            //}

            //m_pov = new CheckBox[4];

            //for (int i = 0; i < 4; i++)
            //{
            //    Size size = new Size(panel2.Width / 4, panel2.Height);

            //    m_pov[i] = new CheckBox();
            //    m_pov[i].Tag = i;

            //    m_pov[i].Location = new Point(i * size.Width, 0);
            //    m_pov[i].Size = size;

            //    m_pov[i].Appearance = Appearance.Button;

            //    m_pov[i].Text = String.Format("POV{0} Up", i);

            //    m_pov[i].CheckedChanged += new EventHandler(Pov_CheckedChanged);

            //    panel2.Controls.Add(m_pov[i]);
            //}
        }

        private void trkXAxis_Scroll(object sender, EventArgs e)
        {
            joystick.SetAxis((int)(((double)trkXAxis.Value*32767)/511), id, HID_USAGES.HID_USAGE_Y);
            //m_vjoy.SetXAxis(0, (short)(((double)trkXAxis.Value/255)*32767-1));
            //m_vjoy.Update(0);
        }

        private void trkYAxis_Scroll(object sender, EventArgs e)
        {
            joystick.SetAxis((trkYAxis.Value), id, HID_USAGES.HID_USAGE_X);
            //m_vjoy.SetYAxis(0, (short)trkYAxis.Value);
            //m_vjoy.Update(0);
        }

    
        private void timer1_Tick(object sender, EventArgs e)
        {
            //m_vjoy.SetXAxis(0, (short)(((double)trkXAxis.Value / 255) * 32767 - 1));
            //m_vjoy.SetYAxis(0, (short)trkYAxis.Value);
            //m_vjoy.Update(0);
        }

    }
}
