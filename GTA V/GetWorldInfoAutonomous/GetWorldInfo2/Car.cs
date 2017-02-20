using GTA.Math;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GTA;
using DeveloperConsole;

namespace GetWorldInfo
{

    [Serializable]
    public class Car
    {
        public int Handle { get; set; }


        public Vector3 Position;

        public Vector3 Rotation;
        public Car()
        {

        }
        public Car(Vehicle v,List<Point> screenBounds,float distance,Point CenterCam)
        {
            AccInput = ((Vehicle)v).Acceleration;
            CarType = ((Vehicle)v).ClassType.ToString();
            CenterCamPosition = CenterCam;
            DistanceToCam = distance;
            Gear = ((Vehicle)v).CurrentGear;
            Handle = v.Handle;
            Position = v.Position;
            Rotation = v.Rotation;
            RPM = ((Vehicle)v).CurrentRPM;
            ScreenBounds =screenBounds;
            Speed = ((Vehicle)v).Speed;
            SteeringAngle = ((Vehicle)v).SteeringAngle;
            WheelSpeed = ((Vehicle)v).WheelSpeed;
            var size = v.Model.GetDimensions();
            var location = v.Position - (size / 2);
            Rectangle3D rect = new Rectangle3D(location, size).Rotate(GTAFuncs.GetEntityQuaternion(v));
            Bounds = (from p in rect.Corners.ToList() select p.Value).ToList();
        }

        public List<Vector3> Bounds { get; set; }
        public Point CenterCamPosition { get; set; }
        public List<Point> ScreenBounds { get; set; }
        public float DistanceToCam { get; set; }
        public float Speed { get; set; }
        public float AccInput { get; set; }
        public float RPM { get; set; }
        public int Gear { get; set; }
        public float SteeringAngle { get; set; }
        public float WheelSpeed { get; set; }
        public string CarType { get; set; }
    }
}
