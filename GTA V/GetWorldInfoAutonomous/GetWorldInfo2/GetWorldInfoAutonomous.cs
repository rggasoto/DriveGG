using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Windows.Forms;
using GTA;
using GTA.Math;
using GTA.Native;
using System.Drawing;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Xml.Serialization;
using DeveloperConsole;
using GetWorldInfo;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Reflection;
using Emgu.CV;
using Emgu.CV.Structure;

namespace GetCarInfo
{


    public class GetWorldInfoAutonomous : Script
    {
        Vector3 offset;
        Rectangle SysRes;
        Rectangle gameScreen;
        string Fileformat;
        bool Save = false;
        Socket clientSocket;
        IPHostEntry hostInfo;
        IPAddress serverAddr;
        const string DEFAULT_SERVER = "localhost";
        const int DEFAULT_PORT = 809;
        Matrix<double> Warp;
        void Log(string logLevel, List<string> message)
        {
            DateTime datetime = DateTime.Now;
            string logPath = "C:\\Program Files\\Rockstar Games\\Grand Theft Auto V\\GetWorldInfoAutonomous.log";

            try
            {
                var fs = new System.IO.FileStream(logPath, FileMode.Append, FileAccess.Write, FileShare.Read);
                var sw = new System.IO.StreamWriter(fs);

                try
                {
                    sw.Write(string.Concat("[", datetime.ToString("HH:mm:ss"), "] ", logLevel, " "));

                    foreach (string s in message)

                    {
                        sw.Write(s);
                    }

                    sw.WriteLine();
                }
                finally
                {
                    sw.Close();
                    fs.Close();
                }
            }
            catch (Exception exc) {
            }
        }
        public GetWorldInfoAutonomous()
        {
            offset = new Vector3();
            this.Tick += onTick;
            this.KeyUp += onKeyUp;
            this.KeyDown += onKeyDown;
            SysRes = Screen.PrimaryScreen.Bounds;
            gameScreen = new Rectangle((SysRes.Width - Game.ScreenResolution.Width) / 2, (SysRes.Height - Game.ScreenResolution.Height) / 2, Game.ScreenResolution.Width, Game.ScreenResolution.Height);
            hostInfo = Dns.GetHostEntry(DEFAULT_SERVER);
            serverAddr = hostInfo.AddressList[1];
            var clientEndPoint = new IPEndPoint(serverAddr, DEFAULT_PORT);
            clientSocket = new System.Net.Sockets.Socket(System.Net.Sockets.AddressFamily.InterNetwork, System.Net.Sockets.SocketType.Stream, System.Net.Sockets.ProtocolType.Tcp);
            try { clientSocket.Connect(clientEndPoint); }
            catch { }
            Log("[DEBUG]", new List<string> { "test" });
            
            PointF[] src = new PointF[4];
            src[0] = new PointF(-300.0f,620.0f);
            src[1] = new PointF(1580.0f,620.0f);
            src[2] = new PointF(736.0f,378.0f);
            src[3] = new PointF(544.0f,378.0f);
            PointF[] dst = new PointF[4];            
            dst[0] = new PointF(0.0f ,720.0f);
            dst[1] = new PointF(1280.0f,720.0f);
            dst[2] = new PointF(1280.0f,0.0f);
            dst[3] = new PointF(0.0f,0.0f);
            try
            {
                Mat w = CvInvoke.GetPerspectiveTransform(src, dst);
               
                Warp = new Matrix<double>(w.Width, w.Height);
                w.ConvertTo(Warp, Emgu.CV.CvEnum.DepthType.Cv64F);
            }
            catch (Exception exc)
            {
                Log("[WARP TRANSFORM ERROR]", new List<string> { exc.Message, exc.InnerException.Message });
            }
            List<string> vals = new List<string>();
            for (int i = 0; i < Warp.Width; i++)
            {
                for (int j = 0; j < Warp.Height; j++)
                {
                    vals.Add(Warp.Data[i,j].ToString());
                }
            }
            Log("W mat", vals);
            //Log("[WARP Matrix]", new List<string> { Warp.Data[0,2].ToString()});
            Interval = 1;
        }
        //public List<Vector2>> GetCalibrationPoints()
        //{
        //    Vector3 cameraPos = GameplayCamera.Position;
        //    Vector3 cameraDir = GameplayCamera.Direction;
        //    List <Vector2> points = new List<Tuple<Vector3, Vector2>>();
        //    for (int i =0; i< 10; i++)
        //    {
        //        for (int j = 0; j< 10; j++)
        //        {
        //            for int k = 0;
        //        }
        //    }
        //}
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                
                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                                        
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
        private void onKeyDown(object sender, KeyEventArgs e)
        {

        }

        private void onKeyUp(object sender, KeyEventArgs e)
        {

            updateCameraLocation(e);
            if (e.KeyCode == Keys.R)
            {
                Save = !Save;
            }
            if (Save)
            {

                Fileformat = "E:\\BehaviorClone"; //+ DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss");
                System.IO.Directory.CreateDirectory(Fileformat);
                Fileformat += "\\{0}{1}";
                
            }

        }
        void updateCameraLocation(KeyEventArgs e)
        {
            float delta = 0.1f;
            if ((e.Modifiers & (Keys.Control | Keys.Alt)) == (Keys.Control | Keys.Alt))
            {
                if (e.KeyCode == Keys.NumPad8)
                {
                    offset.Z += delta;
                }
                if (e.KeyCode == Keys.NumPad2)
                {
                    offset.Z -= delta;
                }
                if (e.KeyCode == Keys.NumPad4)
                {
                    offset.X -= delta;
                }
                if (e.KeyCode == Keys.NumPad6)
                {
                    offset.X += delta;
                }
                if (e.KeyCode == Keys.NumPad9)
                {
                    offset.Y += delta;
                }
                if (e.KeyCode == Keys.NumPad3)
                {
                    offset.Y -= delta;
                }
            }

        }

        private void onTick(object sender, EventArgs e)
        {

            Bitmap bmp;            
            Graphics g;
            bmp = new Bitmap(gameScreen.Width, gameScreen.Height, PixelFormat.Format32bppArgb);
            g = Graphics.FromImage(bmp);            
            int steering = GTA.Game.GetControlValue(1, GTA.Control.VehicleMoveLeft);
            int acc = GTA.Game.GetControlValue(1, GTA.Control.VehicleAccelerate);
            int brake = GTA.Game.GetControlValue(1, GTA.Control.VehicleBrake);

            Ped p = Game.Player.Character;
            List<Entity> Vs = (from v in World.GetNearbyVehicles(World.RenderingCamera.Position, 100)
                               where
                                v != p.CurrentVehicle &&
                                v.IsOnScreen &&
                                v.IsVisible
                               select (Entity)v).ToList();
            Vs.AddRange((from v in World.GetNearbyPeds(World.RenderingCamera.Position, 100)
                         where
                          v != p.CurrentVehicle &&
                          v.IsOnScreen &&
                          v.IsVisible &&
                          !v.IsAttached()
                         select (Entity)v).ToList());
            List<Entity> Vs2 = new List<Entity>();
            List<Pedestrian> Peds = new List<Pedestrian>();
            List<Car> Vehicles = new List<Car>();
            //Vector3 d = GTAFuncs.RotationToDirection(World.RenderingCamera.Rotation);
            //Vector3 m = new Vector3(d.X * 1000, d.Y * 1000, d.Z * 1000);
            RaycastResult r;// = World.Raycast(World.RenderingCamera.Position, World.RenderingCamera.Position + d * 200, IntersectOptions.Everything);

            //if (r.DitHitAnything)
            //{
            //    Vector2 P = GTAFuncs.WorldToScreen(r.HitCoords);
            //    float distance = (World.RenderingCamera.Position - r.HitCoords).Length();
            //    UIText statusText = new UIText(string.Format("{0}", brake), new Point((int)P.X, (int)P.Y), 0.2f, Color.FromArgb((int)(255 - 255 * (distance / 200)), (int)(255 * (distance / 200)), 0), GTA.Font.ChaletLondon, false, true, false);
            //    statusText.Draw();
            //}
            //else
            //{
            //    Vector2 P = GTAFuncs.WorldToScreen(r.HitCoords);
            //    UIText statusText = new UIText(string.Format("{0}", r.HitCoords), new Point(2, 2), 0.2f, Color.Red, GTA.Font.ChaletLondon, false, true, false);
            //    statusText.Draw();
            //}
            //Parallel.ForEach(Vs, (v) =>
            foreach (Entity v in Vs)
            {
                float x = 0, y = 0;
                r = World.Raycast(World.RenderingCamera.Position, World.RenderingCamera.Position + (v.Position - World.RenderingCamera.Position).Normalized * 1000, IntersectOptions.Everything);
                #region Check if raycast hit desired object
                if (r.DitHitAnything)
                {
                    if (r.DitHitEntity)
                    {
                        if (r.HitEntity.Equals(v))
                        {
                            if (getScreenPoint(r.HitCoords, ref x, ref y))
                            {
                                Vs2.Add(v);
                                float distance = (World.RenderingCamera.Position - r.HitCoords).Length();
                                //(v.Position - World.RenderingCamera.Position).Length() - distance,
                                //UIText statusText = new UIText(string.Format("{0}",Vs2.IndexOf(v)),  new Point((int)x, (int)y), 0.2f, Color.FromArgb((int)(255 - 255 * (distance / 200)), (int)(255 * (distance / 200)), 0), GTA.Font.ChaletLondon, false, true, false);
                                //DrawEntBox(v, Color.FromArgb((int)(255 - 255 * ((v.Position - World.RenderingCamera.Position).Length() / 200)), (int)(255 * ((p.Position - World.RenderingCamera.Position).Length() / 200)), 0));
                                //statusText.Draw();
                                getScreenPoint(v.Position, ref x, ref y);
                                if (v.GetType().Equals(typeof(Vehicle)))
                                {
                                    #region Add Vehicle
                                    Vehicles.Add(new Car((Vehicle)v, getScreenBounds(v), distance, new Point((int)x, (int)y)));
                                    #endregion                                    
                                }
                                else
                                {
                                    #region Add Pedestrian
                                    Peds.Add(new Pedestrian
                                    {
                                        CenterCamPosition = new Point((int)x, (int)y),
                                        DistanceToCam = distance,
                                        Handle = v.Handle,
                                        Position = v.Position,
                                        ScreenBounds = getScreenBounds(v)
                                    });
                                    #endregion
                                }
                            }
                        }

                    }
                }
                #endregion

            }//);
            Screenshot ss = new Screenshot
            {
                Acc = acc,
                Brake = brake,
                Cars = Vehicles,
                Peds = Peds,
                Position = World.RenderingCamera.Position,
                Rotation = World.RenderingCamera.Rotation,
                Steering = steering,
                Time = Game.GameTime,
                LastFrameTime = Game.LastFrameTime,
                Weather = World.Weather.ToString(),
                DayTime = World.CurrentDayTime.ToString(),
                MyCar = new Car(Game.Player.LastVehicle, new List<Point>(), 0, new Point((int)0, (int)0))

            };
            g.CopyFromScreen(gameScreen.Left, gameScreen.Top, 0, 0, bmp.Size, CopyPixelOperation.SourceCopy);

            if (Save)
            {
                try
                {

                    Image<Emgu.CV.Structure.Bgra, double> original = new Image<Emgu.CV.Structure.Bgra, double>(bmp);
                    var warped = original.WarpPerspective<double>(Warp, Emgu.CV.CvEnum.Inter.Cubic, Emgu.CV.CvEnum.Warp.Default, Emgu.CV.CvEnum.BorderType.Default, new Emgu.CV.Structure.Bgra(0, 0, 0, 0));
                    //CvInvoke.WarpPerspective(original, warped, Warp, bmp.Size, Emgu.CV.CvEnum.Inter.Cubic);
                    Thread ts = new Thread(() => save(ResizeImage(warped.Bitmap, 64, 64), ss));
                    ts.Start();
                }
                catch (Exception exc)
                {
                    Log("[DEBUG]", new List<string> { exc.Message, exc.InnerException.Message });
                }
               
            }
            else
            {
                UIText text = new UIText(Assembly.GetExecutingAssembly().Location, new Point(UI.WIDTH / 2, 15), 0.4f, Color.Blue, GTA.Font.ChaletLondon, true);
                text.Draw();
                Draw(Vs2.ToArray(), World.RenderingCamera.Position);
            }
        }



        private List<Point> getScreenBounds(Entity e)
        {
            var size = e.Model.GetDimensions();
            var location = e.Position - (size / 2);
            Rectangle3D rect = new Rectangle3D(location, size).Rotate(GTAFuncs.GetEntityQuaternion(e));
            List<Vector3> points = (from p in rect.Corners.ToList() select p.Value).ToList();
            List<Point> screenPoints = new List<Point>();
            foreach (Vector3 p in points)
            {
                float x = 0, y = 0;
                if (getScreenPoint(p, ref x, ref y))
                {
                    screenPoints.Add(new Point((int)x, (int)y));
                }
            }
            return screenPoints;
        }

        void Draw(Entity[] es, Vector3 p)
        {
            foreach (Entity e in es)
            {

                float x;
                float y;
                x = 0;
                y = 0;
                if (getScreenPoint(e.Position, ref x, ref y))
                {
                    int health = e.Health;
                    float dist = World.GetDistance(e.Position, p);
                    Type tp = e.GetType();
                    if (tp.Equals(typeof(Vehicle)))
                    {
                        DrawEntBox(e, Color.AliceBlue);
                        UIText statusText = new UIText(string.Format("{0:F2}\n{1:F2} \n{2:F2} \n{3}\n", dist, ((Vehicle)e).Acceleration, ((Vehicle)e).Speed, ((Vehicle)e).ClassType.ToString()), new Point((int)Math.Round(x * UI.WIDTH), (int)Math.Round(y * UI.HEIGHT)), 0.2f, Color.AliceBlue, GTA.Font.ChaletLondon, false, true, false);
                        statusText.Draw();
                    }
                    else if (tp.Equals(typeof(Ped)))
                    {
                        DrawEntBox(e, Color.ForestGreen);
                        UIText statusText = new UIText(string.Format("{0:F2}", dist), new Point((int)Math.Round(x * UI.WIDTH), (int)Math.Round(y * UI.HEIGHT)), 0.2f, Color.ForestGreen, GTA.Font.ChaletLondon, false, true, false);
                        statusText.Draw();
                    }
                    else if (tp.Equals(typeof(Prop)))
                    {
                        switch ((uint)e.Model.Hash)
                        {
                            case 0x3E2B73A4: // prop_traffic_01a
                            case 0x336E5E2A: // prop_traffic_01b
                            case 0xD8EBA922: // prop_traffic_01d
                            case 0xD4729F50: // prop_traffic_02a
                            case 0x272244B2: // prop_traffic_02b
                            case 0x33986EAE: // prop_traffic_03a
                            case 0x2323CDC5: // prop_traffic_03b

                                // Function.Call<bool>(Hash.SET_ENTITY_TRAFFICLIGHT_OVERRIDE, e, 0);
                                DrawEntBox(e, Color.Red);
                                DrawLightBox(e);
                                UIText statusText = new UIText(string.Format("{0}\n{1}\n{2}\n{3}", dist, formatVector(offset), formatVector(e.ForwardVector), formatVector(e.UpVector)), new Point((int)Math.Round(x), (int)Math.Round(y)), 0.4f, Color.MediumBlue, GTA.Font.ChaletLondon, false, true, false);
                                statusText.Draw();
                                break;
                        }



                    }
                }

            }
        }
        byte[] makeoutput(Image bmp, Screenshot ss)
        {
            MemoryStream ms = new MemoryStream();
            bmp.Save(ms, ImageFormat.Png);
            string image = Convert.ToBase64String(ms.ToArray());
            byte[] output = Encoding.UTF8.GetBytes(string.Format("{0};{1};<EOM>", ss.MyCar.Speed, image));
            return output;
        }
        void save(Image bmp, Screenshot ss)
        {
            var output = makeoutput(bmp, ss);
            try
            {
                clientSocket.Send(output);
            }
            catch {
                var clientEndPoint = new IPEndPoint(serverAddr, DEFAULT_PORT);
                try {
                    clientSocket.BeginConnect(clientEndPoint, ac =>
               {
                   clientSocket.Send(output);
               },null);
                }
                catch { }
            }
            //bmp.Save(string.Format(this.Fileformat, ss.Time, ".png"), ImageFormat.Png);
            //XmlSerializer xs = new XmlSerializer(ss.GetType());
            //StreamWriter sw = new StreamWriter(string.Format(this.Fileformat, ss.Time, ".xml"));
            //xs.Serialize(sw, ss);
            //sw.Close();
            //StreamWriter sw = new StreamWriter(string.Format(this.Fileformat, ss.Time, ".xml"));
            //xs.Serialize(sw, ss);
        }
        public void DrawLightBox(Entity e)
        {
            //Distance from base to trafic light
            Vector3 off = new Vector3(Vector3.Dot(e.ForwardVector, offset), Vector3.Dot(e.RightVector, offset), Vector3.Dot(e.UpVector, offset));
            Rectangle3D rect = new Rectangle3D(e.Position + off, new Vector3(.7f, .1f, 1.3f)).Rotate(GTAFuncs.GetEntityQuaternion(e));
            rect.DrawWireFrame(Color.Blue);

        }
        /// <summary>
        ///     Draws a box around the specified entity
        /// </summary>
        /// <param name="e">The entity to draw around</param>
        /// <param name="c">The box color</param>
        public void DrawEntBox(Entity e, Color c)
        {
            var size = e.Model.GetDimensions();
            var location = e.Position - (size / 2);
            new Rectangle3D(location, size).Rotate(GTAFuncs.GetEntityQuaternion(e)).DrawWireFrame(c, true);
        }
        string formatVector(Vector3 v)
        {
            return string.Format("{0:F2};{1:F2};{2:F2}", v.X, v.Y, v.Z);
        }
        unsafe bool getScreenPoint(Vector3 v, ref float x, ref float y)
        {
            float[] vals = new float[2] { 0, 0 };
            fixed (float* xx = &vals[0])
            {
                fixed (float* yy = &vals[1])
                {
                    if (Function.Call<bool>(Hash._WORLD3D_TO_SCREEN2D, v.X, v.Y, v.Z, xx, yy))
                    {

                        x = *xx;
                        y = *yy;
                        x = x * UI.WIDTH;
                        y = y * UI.HEIGHT;
                        return true;
                    }

                    return false;
                }
            }
        }


    }
}

