package org.tensorflow.lite.examples.imagesegmentation;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.EditText;
import android.widget.ImageView;

import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.api.client.extensions.android.http.AndroidHttp;
import com.google.api.client.googleapis.extensions.android.gms.auth.GoogleAccountCredential;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.drive.Drive;
import com.google.api.services.drive.DriveScopes;

import java.util.Collections;


public class BrowseGoogleDrive extends AppCompatActivity {

    private DriveServiceHelper mDriveServiceHelper;
    private String mOpenFileId;
    private static final int REQUEST_CODE_OPEN_DOCUMENT = 2;
    private EditText mFileTitleEditText;
    private ImageView mImageView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_browse_google_drive);
        mFileTitleEditText = findViewById(R.id.file_title_edittext);
        mImageView=findViewById(R.id.iv_image);
        if(getIntent()!=null && getIntent().getExtras()!=null){
            GoogleSignInAccount account= (GoogleSignInAccount) getIntent().getExtras().get("SignedInAccount");

            GoogleAccountCredential credential =
                    GoogleAccountCredential.usingOAuth2(
                            this, Collections.singleton(DriveScopes.DRIVE_FILE));
            credential.setSelectedAccount(account.getAccount());
            Drive googleDriveService =
                    new Drive.Builder(
                            AndroidHttp.newCompatibleTransport(),
                            new GsonFactory(),
                            credential)
                            .setApplicationName("Browse Drive")
                            .build();
            // The DriveServiceHelper encapsulates all REST API and SAF functionality.
            // Its instantiation is required before handling any onClick actions.
            mDriveServiceHelper = new DriveServiceHelper(googleDriveService);
            openFilePicker();
        }

    }

    private void openFilePicker() {
        if (mDriveServiceHelper != null) {
            Log.d("BrowseGoogleDrive", "Opening file picker.");

            Intent pickerIntent = mDriveServiceHelper.createFilePickerIntent();

            // The result of the SAF Intent is handled in onActivityResult.
            startActivityForResult(pickerIntent, REQUEST_CODE_OPEN_DOCUMENT);
        }
    }

    private void openFileFromFilePicker(Uri uri) {
        if (mDriveServiceHelper != null) {
            Log.d("BrowseGoogleDrive", "Opening " + uri.getPath());

            mDriveServiceHelper.openFileUsingStorageAccessFramework(getContentResolver(), uri)
                    .addOnSuccessListener(nameAndContent -> {
                        String name = nameAndContent.first;
                        Bitmap image = nameAndContent.second;

                        mFileTitleEditText.setText(name);
                        mImageView.setImageBitmap(image);
                        // Files opened through SAF cannot be modified.
                        setReadOnlyMode();
                    })
                    .addOnFailureListener(exception ->
                            Log.e("BrowseGoogleDrive", "Unable to open file from picker.", exception));
        }
    }

    private void readFile(String fileId) {
        if (mDriveServiceHelper != null) {
            Log.d("BrowseGoogleDrive", "Reading file " + fileId);

            mDriveServiceHelper.readFile(fileId)
                    .addOnSuccessListener(nameAndContent -> {
                        String name = nameAndContent.first;
                        String content = nameAndContent.second;

                        mFileTitleEditText.setText(name);

                        setReadWriteMode(fileId);
                    })
                    .addOnFailureListener(exception ->
                            Log.e("BrowseGoogleDrive", "Couldn't read file.", exception));
        }
    }
    private void setReadWriteMode(String fileId) {
        mFileTitleEditText.setEnabled(true);
        mOpenFileId = fileId;
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent resultData) {
        switch (requestCode) {
            case REQUEST_CODE_OPEN_DOCUMENT:
                if (resultCode == Activity.RESULT_OK && resultData != null) {
                    Uri uri = resultData.getData();
                    if (uri != null) {
                        openFileFromFilePicker(uri);
                    }
                }
                break;
        }

        super.onActivityResult(requestCode, resultCode, resultData);
    }


    private void setReadOnlyMode() {
        mFileTitleEditText.setEnabled(false);
        mOpenFileId = null;
    }
}
