/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com			
			(C) Alexander Blade 2015
*/

#include "script.h"
#include "utils.h"

int idTextureSpeedoBack, idTextureSpeedoArrow;
Cam cam;
Vector3 offset, roffset;
int create_texture(std::string name)
{
	std::string path = GetCurrentModulePath(); // includes trailing slash
	return createTexture((path + name).c_str());
}

void create_textures()
{	
	idTextureSpeedoBack  = create_texture("NativeSpeedoBack.png");
	idTextureSpeedoArrow = create_texture("NativeSpeedoArrow.png");
}

void draw_speedo(float speed, float alpha)
{
	float rotation = speed * 2.51f /*as miles*/ * 1.6f /*as kilometers*/ / 320.0f /*circle max*/ + 0.655f /*arrow initial rotation*/;
	float screencorrection = GRAPHICS::_GET_SCREEN_ASPECT_RATIO(FALSE);
	drawTexture(idTextureSpeedoBack, 0, -9999, 100, 0.2f, 0.2f, 0.5f, 0.5f, 0.9f, 0.9f, 0.0f, screencorrection, 1.0f, 1.0f, 1.0f, alpha);
	drawTexture(idTextureSpeedoArrow, 0, -9998, 100, 0.25f, 0.25f, 0.5f, 0.5f, 0.9f, 0.9f, rotation, screencorrection, 1.0f, 1.0f, 1.0f, alpha);
}

float speedoAlpha;
void draw_text(std::string caption, float x, float y,float text_scale = 1)
{
	// default values
	int text_col[4] = { 255, 255, 255, 255 };
	int font = 0;

	// correcting values for active line


	int screen_w, screen_h;
	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);

	

	// this is how it's done in original scripts

	// text upper part
	UI::SET_TEXT_FONT(font);
	UI::SET_TEXT_SCALE(0.0, text_scale);
	UI::SET_TEXT_COLOUR(text_col[0], text_col[1], text_col[2], text_col[3]);
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	UI::SET_TEXT_EDGE(0, 0, 0, 0, 0);
	UI::_SET_TEXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)caption.c_str());
	UI::_DRAW_TEXT(x,y);

	// text lower part
	UI::SET_TEXT_FONT(font);
	UI::SET_TEXT_SCALE(0.0, text_scale);
	UI::SET_TEXT_COLOUR(text_col[0], text_col[1], text_col[2], text_col[3]);
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	UI::SET_TEXT_EDGE(0, 0, 0, 0, 0);
	UI::_SET_TEXT_GXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)caption.c_str());
	int num25 = UI::_0x9040DFB09BE75706(x,y);

}
void update()
{
	Player player = PLAYER::PLAYER_ID();
	Ped playerPed = PLAYER::PLAYER_PED_ID();

	//// check if player ped exists and control is on (e.g. not in a cutscene)
	//if (!ENTITY::DOES_ENTITY_EXIST(playerPed) || !PLAYER::IS_PLAYER_CONTROL_ON(player))
	//	CAM::RENDER_SCRIPT_CAMS(false, 0, 3000, 1, 0);
	//	return;

	//// check for player ped death and player arrest
	//if (ENTITY::IS_ENTITY_DEAD(playerPed) || PLAYER::IS_PLAYER_BEING_ARRESTED(player, TRUE))
	//	CAM::RENDER_SCRIPT_CAMS(false, 0, 3000, 1, 0);
	//	return;

	//// check if player is in a vehicle and vehicle name isn't being displayed as well as player's phone
	//const int HUD_VEHICLE_NAME = 6;
	if (!PED::IS_PED_IN_ANY_VEHICLE(playerPed, FALSE))
	{
		speedoAlpha = 0.0f;	
		CAM::RENDER_SCRIPT_CAMS(false, 0, 3000, 1, 0);
		return;

	}
	CAM::DETACH_CAM(cam);
	
	
	

	// speed
	float speed = ENTITY::GET_ENTITY_SPEED(PED::GET_VEHICLE_PED_IS_USING(playerPed));
	Vector3 pos = ENTITY::GET_ENTITY_COORDS(PED::GET_VEHICLE_PED_IS_USING(playerPed),TRUE);
	//Attach cam to vehicle with offset
	CAM::ATTACH_CAM_TO_ENTITY(cam, PED::GET_VEHICLE_PED_IS_USING(playerPed), offset.x, offset.y, offset.z, TRUE);
	//set cam rotation equals to vehicle with offset
	Vector3 rot = ENTITY::GET_ENTITY_ROTATION(PED::GET_VEHICLE_PED_IS_USING(playerPed), 0);
	CAM::SET_CAM_ROT(cam, rot.x + roffset.x, rot.y + roffset.y, rot.z + roffset.z, 2); //Last param always 2
	//add road vibration to camera
	CAM::SHAKE_CAM(cam, "ROAD_VIBRATION_SHAKE", speed);	
	//Display cam

	CAM::SET_CAM_ACTIVE(cam, TRUE);
	CAM::RENDER_SCRIPT_CAMS(true, 0, 3000, 1, 0);

	
	
}



void main()
{	
	
	/*create_textures(); */
	offset = Vector3();
	offset.x = 0.0f;
	offset.y = 1.2f;
	offset.z = 0.9f;
		
	roffset = Vector3();
	roffset.x = -0.5f;
	roffset.y = 0.5f;
	roffset.z = 0.0f;

	
	
	cam = CAM::CREATE_CAM("DEFAULT_SCRIPTED_CAMERA", TRUE);
	CAM::SET_CAM_ACTIVE(cam, TRUE);
	CAM::SET_CAM_FOV(cam, 90);

	while (true)
	{
		update();
		WAIT(0);
	}
}

void ScriptMain()
{
	main();
}
