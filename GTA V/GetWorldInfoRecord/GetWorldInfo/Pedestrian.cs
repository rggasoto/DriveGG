using GTA.Math;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GetWorldInfo
{
    [Serializable]
    public class Pedestrian
    {
        public Pedestrian()
        {

        }
        public int Handle { get; set; }

        
        public Vector3 Position { get; set; }

        public Point CenterCamPosition { get; set; }
        public List<Point> ScreenBounds { get; set; }
        public float DistanceToCam { get; set; }
    }
}
